import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# A consistent image transformation for input to the FOPA model
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_image(image_path, device):
    """
    Loads an image as RGB and sends it to the correct device.
    Shape will be (1, 3, 256, 256).
    """
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img)
    return tensor.unsqueeze(0).to(device)


def load_mask(mask_path, device):
    """
    Loads a mask as grayscale and returns (1, 1, 256, 256) tensor on the given device.
    """
    mask = Image.open(mask_path).convert("L")
    tensor = TRANSFORM(mask)
    return tensor.unsqueeze(0).to(device)


def overlay_images(bg_image_pil, fg_image, mask_image, position):
    """
    Pastes the foreground onto the background using the given alpha mask and position.

    Note:
        - All images should be PIL format.
        - bg_image and fg_image must be RGBA.
        - mask should be mode 'L' (grayscale).
    """
    if mask_image.mode != "L":
        mask_image = mask_image.convert("L")

    # Temporary canvas to paste on
    overlay_layer = Image.new("RGBA", bg_image_pil.size, (255, 255, 255, 0))

    if mask_image.size != fg_image.size:
        mask_image = mask_image.resize(fg_image.size, Image.LANCZOS)

    overlay_layer.paste(fg_image, position, mask=mask_image)
    result = Image.alpha_composite(bg_image_pil, overlay_layer)
    return result


# Variation hyperparameters
SCALES = np.linspace(0.05, 2, 5)
ROTATIONS = np.arange(0, 360, 60)

def generate_variations(fg_path, mask_path):
    """
    Creates scaled and rotated versions of a foreground image and its mask.
    Returns:
        List of (rotated_fg_rgb, rotated_mask) tuples.
    """
    fg = Image.open(fg_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    variations = []

    for scale in SCALES:
        for angle in ROTATIONS:
            w, h = fg.size
            new_size = (int(w * scale), int(h * scale))
            fg_scaled = fg.resize(new_size, Image.LANCZOS)
            mask_scaled = mask.resize(new_size, Image.LANCZOS)

            fg_rotated = fg_scaled.rotate(angle, expand=True)
            mask_rotated = mask_scaled.rotate(angle, expand=True)

            # Convert FG to RGB because FOPA expects RGB (not RGBA)
            variations.append((fg_rotated.convert("RGB"), mask_rotated))
    return variations


def get_heatmap(quadrant_image, fg_image, mask_image, model, device):
    """
    Applies the FOPA model to the given patch and variation,
    returning a heatmap indicating placement quality.
    """
    # Resize fg/mask to roughly fit the quadrant
    max_fit = min(quadrant_image.size)
    ratio = max_fit / max(fg_image.size)
    fg_resized = fg_image.resize([int(x * ratio) for x in fg_image.size], Image.LANCZOS)
    mask_resized = mask_image.resize(fg_resized.size, Image.LANCZOS)

    # Convert to tensors (1, C, 256, 256)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    bg_tensor = transform(quadrant_image.convert("RGB")).unsqueeze(0).to(device)
    fg_tensor = transform(fg_resized).unsqueeze(0).to(device)
    mask_tensor = transform(mask_resized).unsqueeze(0).to(device)

    # Forward pass through model
    with torch.no_grad():
        heatmap, _ = model(bg_tensor, fg_tensor, mask_tensor)

    return heatmap.squeeze(0)[0].cpu().numpy()  # Strip batch/channel dim


def split_into_quadrants(image):
    """
    Splits a PIL image into 4 quadrants (as bounding boxes).
    Returns:
        List of 4 tuples: (left, top, right, bottom)
    """
    w, h = image.size
    return [
        (0, 0, w // 2, h // 2),           # Top-left
        (w // 2, 0, w, h // 2),           # Top-right
        (0, h // 2, w // 2, h),           # Bottom-left
        (w // 2, h // 2, w, h)            # Bottom-right
    ]
