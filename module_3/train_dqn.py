"""DQN training loop wiring Module 2 (env) with Module 3 (networks + rewards)."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module_2_simulation.humanoid_env import HumanoidWalkEnv
from module_3.networks import DiscretizedQNetwork, DiscretizedQNetworkConfig
from module_3.rewards import locomotion_reward


# -----------------------------
# DQN Hyperparameters
# -----------------------------
EPISODES = 1000
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
REPLAY_SIZE = 50000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.999
TARGET_UPDATE = 50


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, size: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=size)

    def push(
        self,
        state: np.ndarray,
        action_bins: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action_bins, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32),
            torch.tensor(np.stack(actions), dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.stack(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# -----------------------------
# Main Training Loop
# -----------------------------
TORQUE_BINS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)


def _select_action(
    policy: DiscretizedQNetwork,
    state: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (torque_values, bin_indices)."""

    num_joints = policy.num_joints
    if random.random() < epsilon:
        bin_indices = np.random.randint(0, len(TORQUE_BINS), size=num_joints, dtype=np.int64)
    else:
        state_tensor = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            logits = policy(state_tensor)
            bin_indices = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
    torques = TORQUE_BINS[bin_indices]
    return torques.astype(np.float32), bin_indices.astype(np.int64)


def train() -> None:
    env = HumanoidWalkEnv(use_gui=False)

    state_dim = env.observation_space.shape[0]
    num_joints = env.action_space.shape[0]

    config = DiscretizedQNetworkConfig(
        state_dim=state_dim,
        num_joints=num_joints,
        torque_bins=TORQUE_BINS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DiscretizedQNetwork(config).to(device)
    target_net = DiscretizedQNetwork(config).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_SIZE)

    eps = EPS_START

    for ep in range(EPISODES):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            torques, bin_indices = _select_action(policy_net, state, eps, device)

            next_state, env_reward, terminated, truncated, info = env.step(torques)
            fell = terminated or truncated
            base_pos = info.get("base_pos", (0.0, 0.0, 0.0))
            torso_height = base_pos[2]
            forward_velocity = env_reward

            shaped_reward = locomotion_reward(
                forward_velocity=forward_velocity,
                torso_height=torso_height,
                min_height=env.termination_height,
                joint_torques=torques,
                fell=fell,
            )

            replay.push(state, bin_indices, shaped_reward, next_state, fell)
            state = next_state
            ep_reward += shaped_reward
            done = fell

            if len(replay) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                q_pred = policy_net(states)
                q_taken = q_pred.gather(2, actions.unsqueeze(-1)).squeeze(-1)
                q_taken = q_taken.mean(dim=1)

                with torch.no_grad():
                    next_q = target_net(next_states)
                    next_q_max = next_q.max(dim=2)[0].mean(dim=1)
                    targets = rewards + GAMMA * next_q_max * (1 - dones)

                loss = nn.MSELoss()(q_taken, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        eps = max(EPS_END, eps * EPS_DECAY)
        print(f"EP {ep} | Reward: {ep_reward:.2f} | EPS: {eps:.3f}")

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print("Target network updated!")

    torch.save(policy_net.state_dict(), "dqn_humanoid.pth")
    print("Training finished! Model saved as dqn_humanoid.pth")

    env.close()


if __name__ == "__main__":
    train()
