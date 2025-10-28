# task1_3_multipeople.py
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from image_io import load_image_rgb  # your existing image loader

# --- Initialize models ---
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 tiny for fast person detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


# --- Functions ---

def extract_skeleton(image_rgb):
    """
    Extract a single person's 33 keypoints [x, y, visibility] using MediaPipe.
    """
    results = pose.process(image_rgb)
    skeleton = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            skeleton.append([lm.x, lm.y, lm.visibility])
        skeleton = np.array(skeleton)
    else:
        skeleton = np.zeros((33, 3))  # fallback if no pose detected

    return skeleton


def multi_person_skeletons(image_bgr):
    """
    Detect multiple people using YOLO and return list of skeletons [num_people, 33, 3].
    """
    results = yolo_model(image_bgr)[0]  # YOLO detections
    skeletons = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) != 0:  # only class 0 = person
            continue

        x1, y1, x2, y2 = map(int, box)
        person_crop = image_bgr[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        # Convert to RGB for MediaPipe
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        sk = extract_skeleton(person_rgb)

        # Convert normalized keypoints back to original image coordinates
        sk[:, 0] = sk[:, 0] * (x2 - x1) + x1
        sk[:, 1] = sk[:, 1] * (y2 - y1) + y1

        skeletons.append(sk)

    return skeletons


def select_main_skeleton(skeletons):
    """
    Choose one skeleton among multiple: largest bounding box area.
    """
    if not skeletons:
        return None

    max_area = 0
    main_sk = None

    for sk in skeletons:
        visible = sk[sk[:, 2] > 0.1]
        if visible.size == 0:
            continue

        x_min, y_min = np.min(visible[:, 0]), np.min(visible[:, 1])
        x_max, y_max = np.max(visible[:, 0]), np.max(visible[:, 1])
        area = (x_max - x_min) * (y_max - y_min)

        if area > max_area:
            max_area = area
            main_sk = sk

    return main_sk


def vector_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.arctan2(dy, dx)


def compute_joint_angles(skeleton):
    """
    Convert skeleton (33,3) to humanoid joint angles (θ_init).
    Example includes legs and right arm.
    """
    # Right leg
    R_HIP, R_KNEE, R_ANKLE = mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value
    θ_hip_r = vector_angle(skeleton[R_HIP], skeleton[R_KNEE])
    θ_knee_r = vector_angle(skeleton[R_KNEE], skeleton[R_ANKLE]) - θ_hip_r

    # Left leg
    L_HIP, L_KNEE, L_ANKLE = mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value
    θ_hip_l = vector_angle(skeleton[L_HIP], skeleton[L_KNEE])
    θ_knee_l = vector_angle(skeleton[L_KNEE], skeleton[L_ANKLE]) - θ_hip_l

    # Right arm
    R_SHOULDER, R_ELBOW, R_WRIST = mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value
    θ_shoulder_r = vector_angle(skeleton[R_SHOULDER], skeleton[R_ELBOW])
    θ_elbow_r = vector_angle(skeleton[R_ELBOW], skeleton[R_WRIST]) - θ_shoulder_r

    θ_init = np.array([θ_hip_r, θ_knee_r, θ_hip_l, θ_knee_l, θ_shoulder_r, θ_elbow_r])
    return θ_init


# --- Main pipeline ---

if __name__ == "__main__":
    img_path = "../data/image_3.png"  # adjust path
    img_rgb = load_image_rgb(img_path)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Detect multiple people
    skeletons = multi_person_skeletons(img_bgr)
    print(f"Detected {len(skeletons)} people")

    # Select main skeleton
    main_skeleton = select_main_skeleton(skeletons)
    if main_skeleton is None:
        print("No person detected!")
    else:
        θ_init = compute_joint_angles(main_skeleton)
        print("Initial Pose Vector θ_init:", θ_init)

        # Optional: visualize main skeleton
        for x, y, _ in main_skeleton:
            cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 255, 0), -1)

        cv2.imshow("Main Person Skeleton", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
