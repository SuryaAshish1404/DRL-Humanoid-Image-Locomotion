from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pybullet as p

from common.config import ControllerConfig


class PoseHoldController:
    """Configurable PD controller that holds joint targets smoothly."""

    def __init__(self, cfg: ControllerConfig, pose_filter_alpha: float, default_force: float) -> None:
        self.cfg = cfg
        self.pose_filter_alpha = pose_filter_alpha
        self.default_force = default_force
        self.humanoid_id: int | None = None
        self.joint_indices: List[int] = []
        self.params: Dict[int, tuple[float, float, float]] = {}
        self.commanded_pose: np.ndarray | None = None

    def setup(
        self,
        humanoid_id: int,
        joint_indices: Sequence[int],
        joint_names: Sequence[str],
        joint_force_limits: Dict[str, float],
    ) -> None:
        self.humanoid_id = humanoid_id
        self.joint_indices = list(joint_indices)
        self.commanded_pose = None
        self.params = {}

        for joint_idx, joint_name in zip(joint_indices, joint_names):
            force, kp, kd = self._params_for_joint(joint_name, joint_force_limits.get(joint_name, self.default_force))
            self.params[joint_idx] = (force, kp, kd)
            p.changeDynamics(self.humanoid_id, joint_idx, linearDamping=0.04, angularDamping=0.04)

    def apply(self, target_positions: np.ndarray) -> np.ndarray:
        if self.humanoid_id is None:
            raise RuntimeError("PoseHoldController.setup must be called before apply")

        targets = np.asarray(target_positions, dtype=np.float32)
        if self.commanded_pose is None or self.commanded_pose.shape != targets.shape:
            self.commanded_pose = targets.copy()
        else:
            alpha = self.pose_filter_alpha
            self.commanded_pose = (1.0 - alpha) * self.commanded_pose + alpha * targets

        for joint_idx, target in zip(self.joint_indices, self.commanded_pose):
            force, kp, kd = self.params.get(joint_idx, (self.default_force, 0.6, 0.1))
            p.setJointMotorControl2(
                self.humanoid_id,
                joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(target),
                targetVelocity=0.0,
                force=force,
                positionGain=kp,
                velocityGain=kd,
            )

        return self.commanded_pose

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _params_for_joint(self, joint_name: str, base_force: float) -> tuple[float, float, float]:
        category = self._category_for_joint(joint_name)
        force = max(base_force, getattr(self.cfg, f"{category}_force"))
        kp = getattr(self.cfg, f"{category}_kp")
        kd = getattr(self.cfg, f"{category}_kd")
        return force, kp, kd

    @staticmethod
    def _category_for_joint(joint_name: str) -> str:
        lower = joint_name.lower()
        if "hip" in lower:
            return "hip"
        if "knee" in lower:
            return "knee"
        if "shoulder" in lower:
            return "shoulder"
        if "elbow" in lower:
            return "elbow"
        return "other"
