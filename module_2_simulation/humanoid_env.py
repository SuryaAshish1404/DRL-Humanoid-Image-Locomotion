"""Custom Gymnasium environment that wraps the humanoid PyBullet simulation."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from common.config import ProjectConfig
from module_1_pose_estimation.pose_estimator import estimate_initial_pose
from module_1_pose_estimation.kinematics import JOINT_NAMES as THETA_NAMES
from .controller import PoseHoldController


class HumanoidWalkEnv(gym.Env):

    """PyBullet + Gymnasium environment for the custom humanoid."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        urdf_path: Optional[str] = None,
        use_gui: bool = False,
    ) -> None:
        super().__init__()

        self.config = config or ProjectConfig.from_file(Path(__file__).resolve().parents[1] / "config" / "project_config.json")
        self.use_gui = use_gui
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.gravity = self.config.simulation.gravity

        self.urdf_path = urdf_path or str(self.config.paths.urdf_path)
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"humanoid.urdf not found at {self.urdf_path}")

        self.action_scale = self.config.simulation.action_scale
        self.termination_height = self.config.simulation.termination_height
        self.pose_filter_alpha = self.config.simulation.pose_filter_alpha
        self.commanded_pose: Optional[np.ndarray] = None
        self.joint_limits: Dict[str, Tuple[float, float]] = {}
        self.joint_force_limits: Dict[str, float] = {}
        self.joint_name_to_index: Dict[str, int] = {}
        self.default_hold_force = self.config.simulation.default_hold_force
        self.controller = PoseHoldController(
            self.config.controller,
            pose_filter_alpha=self.pose_filter_alpha,
            default_force=self.default_hold_force,
        )
        self._load_world()

        self.joint_indices, self.joint_names = self._collect_joints()
        self.default_pose = np.zeros(len(self.joint_indices), dtype=np.float32)
        self.commanded_pose = self.default_pose.copy()
        self.controller.setup(
            self.humanoid_id,
            self.joint_indices,
            self.joint_names,
            self.joint_force_limits,
        )
        self.prev_base_x = 0.0

        obs_dim = len(self.joint_indices) * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.joint_indices),), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        initial_pose: Optional[Union[Sequence[float], Dict[str, float]]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset simulation. Provide initial_pose from Module 1 if available."""

        super().reset(seed=seed)
        if initial_pose is None and options is not None:
            initial_pose = options.get("initial_pose")

        self._load_world()
        self.joint_indices, self.joint_names = self._collect_joints()

        if initial_pose is not None:
            self.default_pose = self._apply_pose(initial_pose)
        else:
            self.default_pose = np.zeros(len(self.joint_indices), dtype=np.float32)

        self.commanded_pose = self.default_pose.copy()
        self.controller.setup(
            self.humanoid_id,
            self.joint_indices,
            self.joint_names,
            self.joint_force_limits,
        )
        self.prev_base_x = p.getBasePositionAndOrientation(self.humanoid_id)[0][0]
        self._snap_to_ground()
        self._settle_pose()
        return self._get_obs(), {}

    def step(
        self, action: Sequence[float]
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply joint targets, step physics, and compute walking reward."""

        if len(action) != len(self.joint_indices):
            raise ValueError(
                f"Expected action of length {len(self.joint_indices)}, got {len(action)}"
            )

        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_positions = self.default_pose + self.action_scale * action

        self.commanded_pose = self.controller.apply(target_positions)

        p.stepSimulation()

        base_pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
        reward = base_pos[0] - self.prev_base_x
        self.prev_base_x = base_pos[0]
        terminated = base_pos[2] < self.termination_height
        truncated = False

        obs = self._get_obs()
        info = {"base_pos": base_pos, "joint_names": self.joint_names}
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_world(self) -> None:
        p.resetSimulation()
        p.setGravity(0, 0, self.gravity)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.humanoid_id = p.loadURDF(
            self.urdf_path, basePosition=[0, 0, 0.8], useFixedBase=False
        )

    def _collect_joints(self) -> Tuple[list[int], list[str]]:
        indices, names = [], []
        self.link_name_to_index = {}
        self.joint_name_to_index: Dict[str, int] = {}
        self.joint_limits = {}
        self.joint_force_limits = {}
        for joint_idx in range(p.getNumJoints(self.humanoid_id)):
            info = p.getJointInfo(self.humanoid_id, joint_idx)
            joint_type = info[2]
            link_name = info[12].decode("utf-8")
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                indices.append(joint_idx)
                joint_name = info[1].decode("utf-8")
                names.append(joint_name)
                self.joint_name_to_index[joint_name] = joint_idx
                lower = info[8]
                upper = info[9]
                max_force = info[10]
                if lower < upper:
                    self.joint_limits[joint_name] = (lower, upper)
                else:
                    self.joint_limits[joint_name] = (-np.inf, np.inf)
                self.joint_force_limits[joint_name] = max_force if max_force > 0 else self.default_hold_force
            self.link_name_to_index[link_name] = joint_idx

        return indices, names

    def _get_obs(self) -> np.ndarray:
        joint_positions = []
        joint_velocities = []
        for idx in self.joint_indices:
            state = p.getJointState(self.humanoid_id, idx)
            joint_positions.append(state[0])
            joint_velocities.append(state[1])
        return np.array(joint_positions + joint_velocities, dtype=np.float32)

    def _apply_pose(self, pose: Union[Sequence[float], Dict[str, float]]) -> np.ndarray:
        if isinstance(pose, dict):
            pose_dict = pose
        else:
            pose_vec = np.asarray(pose, dtype=np.float32)
            if pose_vec.shape[0] == len(self.joint_indices):
                pose_dict = {name: float(val) for name, val in zip(self.joint_names, pose_vec)}
            elif pose_vec.shape[0] == len(THETA_NAMES):
                pose_dict = {name: float(val) for name, val in zip(THETA_NAMES, pose_vec)}
            else:
                raise ValueError(
                    f"Initial pose length {pose_vec.shape[0]} does not match expected {len(self.joint_indices)} or {len(THETA_NAMES)} joints"
                )

        unknown_joints = [name for name in pose_dict if name not in self.joint_names]
        if unknown_joints:
            raise ValueError(
                f"Initial pose referenced unknown joints: {unknown_joints}. Expected one of {self.joint_names}"
            )

        pose_vec = np.array(self.default_pose, copy=True)
        for idx, name in enumerate(self.joint_names):
            if name in pose_dict:
                target = float(pose_dict[name])
                lower, upper = self.joint_limits.get(name, (-np.inf, np.inf))
                if lower < upper:
                    target = float(np.clip(target, lower, upper))
                pose_vec[idx] = target

        for joint_idx, target in zip(self.joint_indices, pose_vec):
            p.resetJointState(self.humanoid_id, joint_idx, targetValue=float(target))

        return pose_vec

    def _snap_to_ground(self) -> None:
        min_z = float("inf")
        for toe in ("right_toe", "left_toe"):
            link_idx = self.link_name_to_index.get(toe)
            if link_idx is None:
                continue
            link_state = p.getLinkState(self.humanoid_id, link_idx)
            if link_state is None:
                continue
            min_z = min(min_z, link_state[0][2])

        if min_z == float("inf"):
            return

        base_pos, base_ori = p.getBasePositionAndOrientation(self.humanoid_id)
        target_z = base_pos[2] - min_z
        target_z = max(target_z, 0.3)
        p.resetBasePositionAndOrientation(self.humanoid_id, [base_pos[0], base_pos[1], target_z], base_ori)

    def _settle_pose(self, steps: Optional[int] = None, soften_gravity: bool = True) -> None:
        """Advance the simulation briefly so the humanoid settles before stepping."""

        steps = steps or self.config.simulation.settle_steps
        original_gravity = self.gravity
        if soften_gravity:
            p.setGravity(0, 0, original_gravity * 0.4)

        for _ in range(steps):
            p.stepSimulation()

        if soften_gravity:
            p.setGravity(0, 0, original_gravity)

    def close(self) -> None:
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)


def demo_run(image_path: Union[str, Path], live: bool) -> None:
    """Manual preview loop. live=True applies random actions, else static pose."""

    env = HumanoidWalkEnv(use_gui=True)

    image_path = Path(image_path)
    try:
        theta_dict, info = estimate_initial_pose(str(image_path), save_debug=False)
        init_pose = theta_dict
        print("Loaded θ_init from", image_path)
        print("Pose info:", info)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to load θ_init from {image_path}: {exc}\nUsing zero pose instead.")
        init_pose = np.zeros(env.action_space.shape[0], dtype=np.float32)

    obs, _ = env.reset(initial_pose=init_pose)
    print("Initial observation:", obs)

    try:
        if live:
            print("Live mode: holding θ_init pose each step. Press Ctrl+C to stop.")
            zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
            while True:
                obs, reward, terminated, truncated, info = env.step(zero_action)
                if terminated or truncated:
                    env._apply_pose(init_pose)
                    env._snap_to_ground()
                time.sleep(1.0 / 240.0)
        else:
            print("Static preview mode. Pose will hold until Ctrl+C.")
            while True:
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Humanoid pose preview")
    parser.add_argument(
        "--image",
        default=str(Path(__file__).resolve().parents[1] / "data" / "isl_3.jpg"),
        help="Path to the image used for θ_init extraction",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Apply random actions instead of holding the static pose",
    )
    args = parser.parse_args()
    demo_run(args.image, live=args.live)
