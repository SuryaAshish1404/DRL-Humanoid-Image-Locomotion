"""DQN training loop wiring Module 2 (env) with Module 3 (networks + rewards)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module_2_simulation.humanoid_env import HumanoidWalkEnv
from module_1_pose_estimation.pose_estimator import estimate_initial_pose
from module_3.rewards import RewardWeights, locomotion_reward


MAX_STEPS = 800
CYCLE_DURATION = 120  # timesteps for one gait cycle
CUSTOM_WEIGHTS = RewardWeights(w_vel=3.0, w_live=0.2, w_energy=0.001, fall_penalty=60.0)
THETA_IMAGE = PROJECT_ROOT / "data" / "image_2.png"


def scripted_torques(step: int, joint_index: dict[str, int], total_joints: int) -> np.ndarray:
    """Generate a simple walking-like joint target pattern over time."""

    torques = np.zeros(total_joints, dtype=np.float32)

    phase = (2 * math.pi * (step % CYCLE_DURATION)) / CYCLE_DURATION
    hip_amp = 0.9
    knee_amp = 1.1
    ankle_amp = 0.6
    toe_amp = 0.4
    shoulder_amp = 0.5
    spine_amp = 0.3

    hip_right = hip_amp * math.sin(phase)
    hip_left = -hip_right
    knee_right = knee_amp * math.sin(phase + math.pi / 2)
    knee_left = -knee_right
    ankle_right = ankle_amp * math.sin(phase + math.pi)
    ankle_left = -ankle_right
    toe_right = toe_amp * math.sin(phase + 3 * math.pi / 2)
    toe_left = -toe_right

    def set_joint(name: str, value: float) -> None:
        idx = joint_index.get(name)
        if idx is not None:
            torques[idx] = value

    set_joint("right_hip", hip_right)
    set_joint("left_hip", hip_left)
    set_joint("right_knee", knee_right)
    set_joint("left_knee", knee_left)
    set_joint("right_ankle", ankle_right)
    set_joint("left_ankle", ankle_left)
    set_joint("right_toe", toe_right)
    set_joint("left_toe", toe_left)

    set_joint("right_shoulder", -shoulder_amp * math.sin(phase))
    set_joint("left_shoulder", shoulder_amp * math.sin(phase))
    set_joint("spine", spine_amp * math.sin(phase / 2))

    return torques


def run_scripted_walk() -> None:
    env = HumanoidWalkEnv(use_gui=True)
    theta_init, _ = estimate_initial_pose(str(THETA_IMAGE))
    state, _ = env.reset(initial_pose=theta_init)

    num_joints = env.action_space.shape[0]
    joint_index = {name: idx for idx, name in enumerate(env.joint_names)}
    rewards: List[float] = []
    cumulative_reward = 0.0

    for step in range(MAX_STEPS):
        torques = scripted_torques(step, joint_index, num_joints)
        next_state, env_reward, terminated, truncated, info = env.step(torques)
        base_pos = info.get("base_pos", (0.0, 0.0, 0.0))
        torso_height = base_pos[2]
        fell = terminated or truncated or torso_height < env.termination_height

        shaped_reward = locomotion_reward(
            forward_velocity=env_reward,
            torso_height=torso_height,
            min_height=env.termination_height,
            joint_torques=torques,
            weights=CUSTOM_WEIGHTS,
            fell=fell,
        )

        cumulative_reward += shaped_reward
        rewards.append(shaped_reward)
        state = next_state

        if fell:
            print(f"Terminated at step {step} (fell).")
            break

    print(f"Scripted walk finished | Steps: {len(rewards)} | Cumulative reward: {cumulative_reward:.2f}")
    print("Reward trace:", rewards)
    print("Press Enter to close the PyBullet window...")
    try:
        input()
    finally:
        env.close()


if __name__ == "__main__":
    run_scripted_walk()
