"""Neural network architectures for Module 3 tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class QNetworkConfig:
    """Configuration for the MLP-based Q-Network."""

    state_dim: int
    action_dim: int
    hidden_layers: Sequence[int] = (256, 256, 256)
    activation: str = "relu"
    dropout: Optional[float] = None


class QNetwork(nn.Module):
    """Multi-layer perceptron Q-network for discrete action spaces."""

    def __init__(self, config: QNetworkConfig) -> None:
        super().__init__()

        if config.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if config.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not config.hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")

        activation_cls = self._resolve_activation(config.activation)

        layers: List[nn.Module] = []
        input_dim = config.state_dim
        for hidden_dim in config.hidden_layers:
            if hidden_dim <= 0:
                raise ValueError("hidden layer sizes must be positive")
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_cls())
            if config.dropout is not None:
                if not 0.0 <= config.dropout < 1.0:
                    raise ValueError("dropout must be in [0, 1)")
                layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, config.action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions given the state batch."""

        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.model(state)

    @staticmethod
    def _resolve_activation(name: str) -> type[nn.Module]:
        name = name.lower()
        if name == "relu":
            return nn.ReLU
        if name == "leaky_relu":
            return nn.LeakyReLU
        if name == "elu":
            return nn.ELU
        if name == "gelu":
            return nn.GELU
        raise ValueError(f"Unsupported activation: {name}")


@dataclass
class DiscretizedQNetworkConfig:
    """Configuration for the multi-head discretized torque Q-network."""

    state_dim: int
    num_joints: int
    torque_bins: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    hidden_layers: Sequence[int] = (256, 256, 256)
    activation: str = "relu"
    dropout: Optional[float] = None


class DiscretizedQNetwork(nn.Module):
    """Q-network whose output head yields per-joint torque-bin logits.

    The network produces `num_joints * len(torque_bins)` outputs that are reshaped
    to `(batch, num_joints, num_bins)`. Selecting the argmax along the last axis
    yields the discrete torque bin for each joint, making it compatible with DQN
    style algorithms that require discrete actions.
    """

    def __init__(self, config: DiscretizedQNetworkConfig) -> None:
        super().__init__()

        if config.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if config.num_joints <= 0:
            raise ValueError("num_joints must be positive")
        torque_bins = list(config.torque_bins)
        if not torque_bins:
            raise ValueError("torque_bins must contain at least one value")
        if not config.hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")

        self.num_joints = config.num_joints
        self.torque_bins = tuple(float(v) for v in torque_bins)
        self.num_bins = len(self.torque_bins)

        activation_cls = QNetwork._resolve_activation(config.activation)

        layers: List[nn.Module] = []
        input_dim = config.state_dim
        for hidden_dim in config.hidden_layers:
            if hidden_dim <= 0:
                raise ValueError("hidden layer sizes must be positive")
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_cls())
            if config.dropout is not None:
                if not 0.0 <= config.dropout < 1.0:
                    raise ValueError("dropout must be in [0, 1)")
                layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim

        output_dim = self.num_joints * self.num_bins
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return logits shaped as (batch, num_joints, num_bins)."""

        if state.ndim == 1:
            state = state.unsqueeze(0)
        logits = self.model(state)
        batch_size = logits.shape[0]
        return logits.view(batch_size, self.num_joints, self.num_bins)

    def greedy_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Return the torque value per joint using argmax over bins."""

        logits = self.forward(state)
        best_idx = logits.argmax(dim=-1)
        bin_values = torch.tensor(self.torque_bins, device=logits.device, dtype=logits.dtype)
        torques = bin_values[best_idx]
        return torques.squeeze(0) if torques.shape[0] == 1 else torques
