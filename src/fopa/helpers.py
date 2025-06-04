import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Refer to the single transform used for FOPA
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_image(image_path: str, device: str):
    """
    Load an RGB image, transform and return a (1 x 3 x 256 x 256) tensor on target device.
    """
    img = Image.open(image_path).convert("RGB")
    img_t = TRANSFORM(img)
    return img_t.unsqueeze(0).to(device)

def load_mask(mask_path: str, device: str):
    """
    Load a grayscale mask, transform and return a (1 x 1 x 256 x 256) tensor on target device.
    """
    mask = Image.open(mask_path).convert("L")
    m_t = TRANSFORM(mask)
    return m_t.unsqueeze(0).to(device)

def overlay_images(bg_image_pil: Image.Image, fg_image: Image.Image, mask_image: Image.Image, position: tuple):
    """
    Paste fg_image (RGBA) onto bg_image_pil (RGBA) at `position` using mask_image as alpha,
    return the combined PIL RGBA image.
    """
    if mask_image.mode != "L":
        mask_image = mask_image.convert("L")

    blank = Image.new("RGBA", bg_image_pil.size, (255, 255, 255, 0))
    if mask_image.size != fg_image.size:
        mask_image = mask_image.resize(fg_image.size, Image.LANCZOS)

    blank.paste(fg_image, position, mask=mask_image)
    combined = Image.alpha_composite(bg_image_pil, blank)
    return combined

# Predefined scales and rotations for variation
SCALES = np.linspace(0.05, 2, 5)
ROTATIONS = np.arange(0, 360, 60)

def generate_variations(fg_path: str, mask_path: str):
    """
    Given a foreground RGBA path and a grayscale mask path, generate a list of
    (fg_rotated_rgb, mask_rotated) tuples at multiple scales/rotations.
    """
    fg_image = Image.open(fg_path).convert("RGBA")
    mask_image = Image.open(mask_path).convert("L")
    variations = []
    for scale in SCALES:
        for rot in ROTATIONS:
            w, h = fg_image.size
            new_size = (int(w * scale), int(h * scale))
            fg_resized = fg_image.resize(new_size, Image.LANCZOS)
            mask_resized = mask_image.resize(new_size, Image.LANCZOS)
            fg_rotated = fg_resized.rotate(rot, expand=True)
            mask_rotated = mask_resized.rotate(rot, expand=True)
            fg_rotated_rgb = fg_rotated.convert("RGB")
            variations.append((fg_rotated_rgb, mask_rotated))
    return variations

def get_heatmap(quadrant_image: Image.Image, fg_image: Image.Image, mask_image: Image.Image, model, device: str):
    """
    Given a PIL quadrant_image, PIL fg_image, and PIL mask_image, compute the heatmap
    by running model(bg, fg, mask). Return a numpy array (heatmap).
    """
    # Resize fg and mask to fit quadrant
    max_size = min(quadrant_image.size)
    ratio = max_size / max(quadrant_image.size)
    fg_scaled = fg_image.resize(
        tuple(int(x * ratio) for x in fg_image.size), Image.LANCZOS
    )
    mask_scaled = mask_image.resize(
        tuple(int(x * ratio) for x in mask_image.size), Image.LANCZOS
    )

    # Convert to tensor
    from torchvision.transforms import Compose, Resize, ToTensor
    transform = Compose([Resize((256, 256)), ToTensor()])
    bg_t = transform(quadrant_image.convert("RGB")).unsqueeze(0).to(device)
    fg_t = transform(fg_scaled).unsqueeze(0).to(device)
    mask_t = transform(mask_scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap, _ = model(bg_t, fg_t, mask_t)
    heatmap = heatmap.squeeze(0)[0]  # (1 x H x W) -> (H x W)
    return heatmap.cpu().numpy()

def split_into_quadrants(image: Image.Image):
    """
    Return a list of four (left, top, right, bottom) boxes that partition the image
    into four equal‐sized quadrants.
    """
    w, h = image.size
    return [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
        (w // 2, h // 2, w, h),
    ]
