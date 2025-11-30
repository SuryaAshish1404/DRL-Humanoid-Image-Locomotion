"""Utilities for selecting a primary skeleton among multiple detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class Detection:
    """Container for a detected person skeleton."""

    skeleton: np.ndarray  # shape (25, 3)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float = 0.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def avg_visibility(self) -> float:
        if self.skeleton.size == 0:
            return 0.0
        return float(np.mean(self.skeleton[:, 2]))

class MainDetectionSelector:
    """Strategy object for choosing the best detection."""

    def __init__(self, strategy: str = "largest_bbox") -> None:
        if strategy != "largest_bbox":
            raise ValueError(f"Unsupported selection strategy: {strategy}")
        self.strategy = strategy

    def select(self, detections: Sequence[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        if self.strategy == "largest_bbox":
            return max(detections, key=lambda det: det.area)
        return None


def select_main_detection(detections: Sequence[Detection]) -> Optional[Detection]:
    """Backward-compatible helper that delegates to MainDetectionSelector."""

    return MainDetectionSelector().select(detections)
