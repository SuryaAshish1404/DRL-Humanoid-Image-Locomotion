from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class PathsConfig:
    yolo_weights: Path
    default_output_dir: Path
    urdf_path: Path


@dataclass
class PoseEstimationConfig:
    min_detection_confidence: float
    bbox_padding: int
    grayscale_atol: float
    save_debug: bool


@dataclass
class SimulationConfig:
    gravity: float
    termination_height: float
    action_scale: float
    pose_filter_alpha: float
    default_hold_force: float
    settle_steps: int


@dataclass
class ControllerConfig:
    hip_force: float
    knee_force: float
    shoulder_force: float
    elbow_force: float
    other_force: float
    hip_kp: float
    hip_kd: float
    knee_kp: float
    knee_kd: float
    shoulder_kp: float
    shoulder_kd: float
    elbow_kp: float
    elbow_kd: float
    other_kp: float
    other_kd: float


@dataclass
class ProjectConfig:
    paths: PathsConfig
    pose_estimation: PoseEstimationConfig
    simulation: SimulationConfig
    controller: ControllerConfig

    @classmethod
    def from_file(cls, path: str | Path) -> "ProjectConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Treat relative paths as relative to the project root (parent of the config dir)
        base_dir = path.resolve().parents[1]

        paths_cfg = PathsConfig(
            yolo_weights=base_dir / raw["paths"]["yolo_weights"],
            default_output_dir=base_dir / raw["paths"]["default_output_dir"],
            urdf_path=base_dir / raw["paths"]["urdf_path"],
        )

        pose_cfg = PoseEstimationConfig(**raw["pose_estimation"])
        sim_cfg = SimulationConfig(**raw["simulation"])
        ctrl_cfg = ControllerConfig(**raw["controller"])
        return cls(paths=paths_cfg, pose_estimation=pose_cfg, simulation=sim_cfg, controller=ctrl_cfg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": {
                "yolo_weights": str(self.paths.yolo_weights),
                "default_output_dir": str(self.paths.default_output_dir),
                "urdf_path": str(self.paths.urdf_path),
            },
            "pose_estimation": self.pose_estimation.__dict__,
            "simulation": self.simulation.__dict__,
            "controller": self.controller.__dict__,
        }
