"""Kinematic helpers to convert MediaPipe skeletons to humanoid joint angles."""

from __future__ import annotations

import math

import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose

# Body-25 subset we keep from MediaPipe's 33 landmarks
MAIN_LANDMARKS = [
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
]

JOINT_NAMES = [
    "right_hip",
    "right_knee",
    "left_hip",
    "left_knee",
    "right_shoulder",
    "right_elbow",
]


def _angle(p_a: np.ndarray, p_b: np.ndarray) -> float:
    """Return planar angle between two points."""

    dx = p_b[0] - p_a[0]
    dy = p_b[1] - p_a[1]
    return math.atan2(dy, dx)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_joint_angles(skeleton: np.ndarray) -> np.ndarray:
    """Return the six θ_init angles that drive the simplified humanoid."""

    mp_idx = mp_pose.PoseLandmark

    def landmark_point(name: mp_pose.PoseLandmark) -> np.ndarray:
        return skeleton[MAIN_LANDMARKS.index(name.value)]

    DOWN = math.pi / 2

    hip_r_img = _angle(landmark_point(mp_idx.RIGHT_HIP), landmark_point(mp_idx.RIGHT_KNEE))
    hip_l_img = _angle(landmark_point(mp_idx.LEFT_HIP), landmark_point(mp_idx.LEFT_KNEE))
    knee_r = _angle(landmark_point(mp_idx.RIGHT_KNEE), landmark_point(mp_idx.RIGHT_ANKLE)) - hip_r_img
    knee_l = _angle(landmark_point(mp_idx.LEFT_KNEE), landmark_point(mp_idx.LEFT_ANKLE)) - hip_l_img
    hip_r = hip_r_img - DOWN
    hip_l = DOWN - hip_l_img

    shoulder_r_img = _angle(landmark_point(mp_idx.RIGHT_SHOULDER), landmark_point(mp_idx.RIGHT_ELBOW))
    elbow_r = _angle(landmark_point(mp_idx.RIGHT_ELBOW), landmark_point(mp_idx.RIGHT_WRIST)) - shoulder_r_img
    shoulder_r = shoulder_r_img - DOWN

    hip_r = _clamp(hip_r, -1.2, 1.2)
    hip_l = _clamp(hip_l, -1.2, 1.2)
    knee_r = _clamp(knee_r, -0.1, 1.7)
    knee_l = _clamp(knee_l, -0.1, 1.7)
    shoulder_r = _clamp(shoulder_r, -1.5, 1.5)
    elbow_r = _clamp(elbow_r, -1.0, 1.5)

    return np.array([hip_r, knee_r, hip_l, knee_l, shoulder_r, elbow_r], dtype=np.float32)


# --- Main pipeline ---

if __name__ == "__main__":
    import os
    import cv2
    import numpy as np
    from image_io import load_image_rgb

    img_path = "./data/image_1.png"
    img_rgb = load_image_rgb(img_path)

    # --- Detect if the image is truly grayscale ---
    if np.allclose(img_rgb[:, :, 0], img_rgb[:, :, 1], atol=2) and np.allclose(img_rgb[:, :, 1], img_rgb[:, :, 2], atol=2):
        print("⚙️ Detected grayscale image – enhancing contrast for better detection.")
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        print("Detected color image – skipping grayscale enhancement.")

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # --- Perform detection ---
    skeletons = multi_person_skeletons(img_bgr)
    print(f"Detected {len(skeletons)} people")

    main_skeleton = select_main_skeleton(skeletons)
    if main_skeleton is None:
        print("No person detected!")
    else:
        θ_init = compute_joint_angles(main_skeleton)
        
        # ✅ Print theta_init
        print("--------------------------------------------------")
        print("Computed Initial Pose Vector θ_init:")
        print(θ_init)
        print("--------------------------------------------------")

        # --- Visualization ---
        for x, y, _ in main_skeleton:
            cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 255, 0), -1)

        # --- Save output image in ../outputs ---
        os.makedirs("../outputs", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join("../outputs", f"{base_name}_kinematics_output.jpg")
        cv2.imwrite(output_path, img_bgr)
        print(f"✅ Output image saved as: {output_path}")

        # --- Optional display ---
        # cv2.imshow("Main Person Skeleton (Reduced)", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
