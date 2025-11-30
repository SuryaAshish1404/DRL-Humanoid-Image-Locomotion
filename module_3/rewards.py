"""Reward shaping utilities for humanoid locomotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RewardWeights:
    """Weights used for the locomotion reward."""

    w_vel: float = 1.0
    w_live: float = 0.1
    w_energy: float = 0.01
    fall_penalty: float = 50.0


def locomotion_reward(
    forward_velocity: float,
    torso_height: float,
    min_height: float,
    joint_torques,
    weights: RewardWeights | None = None,
    fell: bool = False,
) -> float:
    """Compute Rt = w_vel*r_vel + w_live*r_live - w_energy*r_energy.

    Args:
        forward_velocity: torso COM velocity along +x (m/s).
        torso_height: current torso COM height (m).
        min_height: threshold below which the agent is considered fallen.
        joint_torques: iterable of torques applied at this step (N·m).
        weights: optional overrides for reward scaling.
        fell: when True, applies a large terminal penalty.
    """

    weights = weights or RewardWeights()

    r_vel = forward_velocity
    r_live = 1.0 if torso_height >= min_height else 0.0
    r_energy = sum(float(t) ** 2 for t in joint_torques)

    reward = weights.w_vel * r_vel + weights.w_live * r_live - weights.w_energy * r_energy

    if fell:
        reward -= weights.fall_penalty

    return reward
