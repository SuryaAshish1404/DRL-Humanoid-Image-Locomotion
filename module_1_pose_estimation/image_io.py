import cv2
import numpy as np
from PIL import Image
import os  # for robust path handling

import cv2
import numpy as np # <-- You need this import!

def load_image_cv(path):
    try:
        with open(path, 'rb') as f:
            file_data = f.read()
        
        np_arr = np.frombuffer(file_data, np.uint8)
        
    except Exception as e:
        raise FileNotFoundError(f"Failed to read image buffer: {path}. Error: {e}")

    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or could not be decoded: {path}")

    return img_bgr

def load_image_rgb(path):
    img = load_image_cv(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def load_image_pil(path):
    """
    Load image using PIL and return as NumPy array (RGB)
    """
    img = Image.open(path).convert('RGB')
    return np.array(img)

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
