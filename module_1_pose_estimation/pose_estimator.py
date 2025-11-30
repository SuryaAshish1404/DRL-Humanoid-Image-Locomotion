from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from common.config import ProjectConfig
from .image_io import load_image_rgb
from .kinematics import JOINT_NAMES, MAIN_LANDMARKS, compute_joint_angles
from .selection import Detection, MainDetectionSelector


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "project_config.json"
_DEFAULT_PIPELINE: Optional["PosePipeline"] = None


def _resolve_config(config: Union[ProjectConfig, str, Path, None]) -> ProjectConfig:
    if isinstance(config, ProjectConfig):
        return config
    config_path = Path(config) if config is not None else DEFAULT_CONFIG_PATH
    return ProjectConfig.from_file(config_path)


class ImagePreprocessor:
    def __init__(self, config: ProjectConfig):
        self.cfg = config.pose_estimation

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img_rgb = load_image_rgb(image_path)
        if self._is_grayscale(img_rgb):
            print("⚙️ Detected grayscale image – enhancing contrast for better detection.")
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            print("Detected color image – skipping grayscale enhancement.")

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_rgb, img_bgr

    def _is_grayscale(self, img_rgb: np.ndarray) -> bool:
        atol = self.cfg.grayscale_atol
        return bool(
            np.allclose(img_rgb[:, :, 0], img_rgb[:, :, 1], atol=atol)
            and np.allclose(img_rgb[:, :, 1], img_rgb[:, :, 2], atol=atol)
        )


class PersonDetector:
    def __init__(self, config: ProjectConfig) -> None:
        self.cfg = config
        self.yolo = YOLO(str(config.paths.yolo_weights))
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=config.pose_estimation.min_detection_confidence,
        )
        self.bbox_padding = config.pose_estimation.bbox_padding

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self.yolo(image_bgr)[0]
        detections: List[Detection] = []

        boxes = results.boxes.xyxy
        classes = results.boxes.cls
        confs = results.boxes.conf

        for box, cls, conf in zip(boxes, classes, confs):
            if int(cls) != 0:
                continue

            x1, y1, x2, y2 = self._pad_and_clip_bbox(box, image_bgr.shape[:2])
            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = image_bgr[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            skeleton = self._extract_skeleton(person_rgb)
            if skeleton.size == 0:
                continue

            skeleton[:, 0] = skeleton[:, 0] * (x2 - x1) + x1
            skeleton[:, 1] = skeleton[:, 1] * (y2 - y1) + y1

            detections.append(
                Detection(
                    skeleton=skeleton,
                    bbox=(x1, y1, x2, y2),
                    score=float(conf),
                )
            )

        return detections

    def _extract_skeleton(self, image_rgb: np.ndarray) -> np.ndarray:
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return np.zeros((len(MAIN_LANDMARKS), 3))

        skeleton = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in MAIN_LANDMARKS:
                skeleton.append([lm.x, lm.y, lm.visibility])
        return np.array(skeleton)

    def _pad_and_clip_bbox(self, box: Sequence[float], image_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = image_hw
        pad = self.bbox_padding
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        return x1, y1, x2, y2


class PosePipeline:
    def __init__(self, config: ProjectConfig, selector: Optional[MainDetectionSelector] = None) -> None:
        self.config = config
        self.preprocessor = ImagePreprocessor(config)
        self.detector = PersonDetector(config)
        self.selector = selector or MainDetectionSelector()

    def run(
        self,
        image_path: str,
        *,
        save_debug: Optional[bool] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, dict]:
        img_rgb, img_bgr = self.preprocessor.preprocess(image_path)
        detections = self.detector.detect(img_bgr)
        main_det = self.selector.select(detections)

        if main_det is None:
            raise ValueError("No valid person detected in image")

        theta = compute_joint_angles(main_det.skeleton)
        debug_info = {
            "image_path": image_path,
            "bbox": main_det.bbox,
            "avg_visibility": main_det.avg_visibility,
        }

        should_save = save_debug if save_debug is not None else self.config.pose_estimation.save_debug
        if should_save:
            out_dir = Path(output_dir) if output_dir is not None else self.config.paths.default_output_dir
            out_dir = Path(out_dir)
            os.makedirs(out_dir, exist_ok=True)

            overlay = img_bgr.copy()
            for x, y, _ in main_det.skeleton:
                cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 0), -1)

            img_name = Path(image_path).stem
            overlay_path = out_dir / f"{img_name}_pose_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay)

            theta_path = out_dir / f"{img_name}_theta_init.json"
            theta_dict = {name: float(val) for name, val in zip(JOINT_NAMES, theta)}
            with open(theta_path, "w", encoding="utf-8") as f:
                json.dump(theta_dict, f, indent=2)

            debug_info["theta_file"] = str(theta_path)
            debug_info["overlay_file"] = str(overlay_path)

        return theta, debug_info


def _get_default_pipeline() -> "PosePipeline":
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        config = _resolve_config(None)
        _DEFAULT_PIPELINE = PosePipeline(config)
    return _DEFAULT_PIPELINE


def estimate_initial_pose(
    image_path: str,
    *,
    config: Union[ProjectConfig, str, Path, None] = None,
    save_debug: Optional[bool] = None,
    output_dir: Optional[Union[str, Path]] = None,
):
    """Full pipeline entrypoint compatible with previous API."""

    if config is None:
        pipeline = _get_default_pipeline()
    else:
        pipeline = PosePipeline(_resolve_config(config))

    return pipeline.run(image_path, save_debug=save_debug, output_dir=output_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = Path(__file__).resolve().parents[1] / "data" / "image_1.png"

    theta, info = estimate_initial_pose(image_path)
    print("Computed θ_init:")
    for name, angle in zip(JOINT_NAMES, theta):
        print(f"  {name}: {angle:.3f}")
    print("Debug info:", info)
