import cv2
import numpy as np
from PIL import Image

def generate_trimap(mask: np.ndarray, erode_kernel_size: int = 10, dilate_kernel_size: int = 10) -> np.ndarray:
    """
    Given a binary mask (0/255), produce a trimap:
    - 255 inside eroded region,
    - 128 in the intermediate (unknown) band,
    - 0 outside the dilated region.
    """
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    eroded = cv2.erode(mask, erode_kernel, iterations=5)
    dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
    trimap = np.zeros_like(mask)
    trimap[dilated == 255] = 128
    trimap[eroded == 255] = 255
    return trimap

def convert_pixels(gray_image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Given a float‐valued trimap between 0.0/0.5/1.0, 
    and a list of bounding boxes [x1,y1,x2,y2], 
    set any pixel that is '1' in gray_image to 0.5 if it lies inside a box.
    """
    converted = np.copy(gray_image)
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        patch = converted[y1:y2, x1:x2]
        patch[patch == 1] = 0.5
        converted[y1:y2, x1:x2] = patch
    return converted

def blend_with_background(foreground: np.ndarray, alpha: np.ndarray, background_path: str) -> np.ndarray:
    """
    Given foreground RGB array (H x W x 3), an alpha mask (H x W, range [0..1]),
    and a background image file path, return a blended RGB array (uint8).
    """
    bg = Image.open(background_path).convert("RGB")
    bg = bg.resize((foreground.shape[1], foreground.shape[0]))
    fg = foreground.astype(float) / 255.0
    a = alpha[:, :, None]
    bg_np = np.array(bg).astype(float) / 255.0
    blended = a * fg + (1 - a) * bg_np # formula from paper for blending
    blended = (blended * 255).astype(np.uint8)
    return blended
