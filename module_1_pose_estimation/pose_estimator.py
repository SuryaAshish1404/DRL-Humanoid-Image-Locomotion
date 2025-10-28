import cv2
import numpy as np
import mediapipe as mp
from image_io import load_image_rgb
from ultralytics import YOLO

# --- Initialize models ---
# YOLOv8 tiny for fast person detection
yolo_model = YOLO("yolov8n.pt")  # downloads automatically if missing

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- Functions ---
def extract_skeleton(image_rgb):
    """
    Extract a single person's 33 keypoints [x, y, confidence] using MediaPipe.
    """
    results = pose.process(image_rgb)
    skeleton = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:  # all 33 landmarks
            skeleton.append([lm.x, lm.y, lm.visibility])
        skeleton = np.array(skeleton)
    else:
        skeleton = np.zeros((33, 3))  # fallback if no pose detected

    return skeleton


def multi_person_skeletons(image_bgr):
    """
    Detect multiple people and return list of skeletons [num_people, 33, 3].
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


# --- Test the pipeline ---
if __name__ == "__main__":
    img_path = "../data/image_3.png"  # adjust your path
    img_rgb = load_image_rgb(img_path)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    skeletons = multi_person_skeletons(img_bgr)
    print(f"Detected {len(skeletons)} people")

    for idx, sk in enumerate(skeletons):
        print(f"Skeleton {idx} shape:", sk.shape)  # should be (33, 3)
        print(sk)

    # --- Optional visualization ---
    for sk in skeletons:
        for x, y, _ in sk:
            cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.imshow("Multi-Person Pose", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
