import cv2
import numpy as np
from PIL import Image

def generate_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
    """
    Takes a binary mask (0s and 255s) and creates a trimap for matting:
      - 255 where we're confident it's foreground (inner region)
      - 128 for intermediate zone (the band between erosion and dilation)
      - 0 for outside dilated region

    Returns:
        A grayscale image with pixel values {0, 128, 255}
    """
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), dtype=np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)

    inner = cv2.erode(mask, erode_kernel, iterations=5)
    outer = cv2.dilate(mask, dilate_kernel, iterations=5)

    trimap = np.zeros_like(mask, dtype=np.uint8)
    trimap[outer == 255] = 128
    trimap[inner == 255] = 255
    return trimap


def convert_pixels(gray_image, boxes):
    """
    Given a grayscale-like float array where pixels are 0.0, 0.5, or 1.0,
    reduce certainty for 1.0 pixels inside any bounding box region by converting them to 0.5.
    """
    modified = gray_image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        patch = modified[y1:y2, x1:x2]
        patch[patch == 1.0] = 0.5
        modified[y1:y2, x1:x2] = patch
    return modified


def blend_with_background(foreground, alpha, background_path):
    """
    Alpha blends a foreground onto a background loaded from disk.

    Params:
        foreground: RGB uint8 array (H x W x 3)
        alpha: single-channel float array (H x W) in [0, 1]
        background_path: path to background image file

    Returns:
        RGB blended image as uint8 numpy array
    """
    bg = Image.open(background_path).convert("RGB")
    bg_resized = bg.resize((foreground.shape[1], foreground.shape[0]))
    bg_np = np.asarray(bg_resized).astype(np.float32) / 255.0

    fg_np = foreground.astype(np.float32) / 255.0
    alpha_expanded = alpha[:, :, np.newaxis]

    blended = alpha_expanded * fg_np + (1.0 - alpha_expanded) * bg_np
    return (blended * 255).astype(np.uint8)
