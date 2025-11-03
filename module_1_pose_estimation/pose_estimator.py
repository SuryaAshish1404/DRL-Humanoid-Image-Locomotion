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

MAIN_LANDMARKS = [
    0,   # nose
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
    5, 6,    # eyes
    7, 8,    # ears
    9, 10,   # mouth corners
    17, 18,  # index fingers
    19, 20,  # pinky fingers
    21, 22   # thumbs
]


# --- Functions ---
def extract_skeleton(image_rgb):
    """
    Extract a single person's reduced 25 keypoints [x, y, confidence] using MediaPipe.
    """
    results = pose.process(image_rgb)
    skeleton = []

    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in MAIN_LANDMARKS:
                skeleton.append([lm.x, lm.y, lm.visibility])
        skeleton = np.array(skeleton)
    else:
        skeleton = np.zeros((len(MAIN_LANDMARKS), 3))  # fallback if no pose detected

    return skeleton


def multi_person_skeletons(image_bgr):
    """
    Detect multiple people and return list of reduced skeletons [num_people, 25, 3].
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
    import os
    import cv2
    import numpy as np
    from image_io import load_image_rgb

    img_path = "../data/isl_11.jpg"  # adjust your path
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

    for idx, sk in enumerate(skeletons):
        print(f"Skeleton {idx} shape:", sk.shape)
        print(sk)

    # --- Visualization ---
    for sk in skeletons:
        for x, y, _ in sk:
            cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 255, 0), -1)

    # --- Save output image in ../outputs ---
    os.makedirs("../outputs", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join("../outputs", f"{base_name}_pose_estimator_output.jpg")
    cv2.imwrite(output_path, img_bgr)
    print(f"Output image saved as: {output_path}")

    # --- Optional display ---
    # cv2.imshow("Multi-Person Pose (Reduced 25 keypoints)", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
