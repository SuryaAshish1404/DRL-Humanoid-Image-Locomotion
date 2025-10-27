import cv2
import numpy as np
from PIL import Image
import os  # for robust path handling

def load_image_rgb(path):
    """
    Load an image and convert to RGB format (for most pose estimation models).
    """
    img = load_image_cv(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# Quick test when running this file directly
if __name__ == "__main__":
    # Robustly build path relative to this file
    current_dir = os.path.dirname(__file__)  # folder of io.py
    sample_path = os.path.join(current_dir, "..", "data", "image_1.png")
    sample_path = os.path.abspath(sample_path)

    print("Using image path:", sample_path)
    print("Exists?", os.path.exists(sample_path))  # Should be True

    # Load images
    img_bgr = load_image_cv(sample_path)
    img_rgb = load_image_rgb(sample_path)
    img_pil = load_image_pil(sample_path)

    print("OpenCV BGR shape:", img_bgr.shape)
    print("OpenCV->RGB shape:", img_rgb.shape)
    print("PIL->NumPy shape:", img_pil.shape)
