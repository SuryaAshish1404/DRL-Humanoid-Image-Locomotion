import cv2
import numpy as np
from PIL import Image
import os  # for robust path handling

import cv2
import numpy as np # <-- You need this import!

def load_image_cv(path):
    # --- The Unicode-Safe Fix ---
    
    # 1. Read the file into a NumPy array (buffer) using Python's standard path handling
    # The 'rb' flag means read in binary mode.
    try:
        with open(path, 'rb') as f:
            file_data = f.read()
        
        # Convert the binary data to a NumPy array
        np_arr = np.frombuffer(file_data, np.uint8)
        
    except Exception as e:
        # Handle cases where the file itself can't be opened/read by Python
        raise FileNotFoundError(f"Failed to read image buffer: {path}. Error: {e}")

    # 2. Decode the NumPy array buffer using OpenCV
    # cv2.IMREAD_COLOR ensures it's loaded in BGR format
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 3. Check for successful decoding
    if img_bgr is None:
        # This handles cases where the file exists but isn't a valid image format
        raise FileNotFoundError(f"Image not found or could not be decoded: {path}")

    return img_bgr

def load_image_rgb(path):
    """
    Load an image and convert to RGB format (for most pose estimation models).
    """
    img = load_image_cv(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def load_image_pil(path):
    """
    Load image using PIL and return as NumPy array (RGB)
    """
    img = Image.open(path).convert('RGB')
    return np.array(img)

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
