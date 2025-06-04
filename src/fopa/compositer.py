import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .helpers import (
    generate_variations,
    get_heatmap,
    overlay_images,
    split_into_quadrants,
)
from .model_init import load_fopa_model

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

def generate_composite_images(background_dir: str, foreground_dir: str, output_dir: str):
    """
    For up to 100 background images and 4 foreground images, run the full
    MatteAnything inference, generate variations, pick best via the FOPA model,
    and overlay onto background. Display intermediate and final results.
    """
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    background_files = [
        os.path.join(background_dir, f)
        for f in os.listdir(background_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:100]

    foreground_files = [
        os.path.join(foreground_dir, f)
        for f in os.listdir(foreground_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:4]

    # Load FOPA model once
    fopa_model = load_fopa_model("best_weight.pth")

    for bg_path in background_files:
        bg_image = Image.open(bg_path).convert("RGBA")
        quadrants = split_into_quadrants(bg_image)

        random.shuffle(foreground_files)
        available_quadrants = list(range(4))
        random.shuffle(available_quadrants)

        for fg_path in foreground_files:
            if not available_quadrants:
                break
            quadrant_index = available_quadrants.pop()
            quadrant_box = quadrants[quadrant_index]
            quadrant_image = bg_image.crop(quadrant_box)

            # 1) Run MatteAnything inference on fg_path to get alpha & mask
            from matte_anything.inference import run_inference
            start_inf = time.time()
            alpha, _, trimap = run_inference(fg_path)
            end_inf = time.time()
            print(f"Time for MatteAnything inference: {end_inf - start_inf:.2f}s")

            # Save grayscale mask
            mask = (alpha * 255).astype(np.uint8)
            grayscale_mask = Image.fromarray(mask)
            mask_name = os.path.splitext(os.path.basename(fg_path))[0] + "_mask.png"
            mask_path = os.path.join(output_dir, mask_name)
            grayscale_mask.save(mask_path)

            # 2) Generate variations & pick best by FOPA heatmap
            variations = generate_variations(fg_path, mask_path)
            best_variation = None
            best_value = -np.inf

            start_var = time.time()
            for fg_var, mask_var in variations:
                heatmap_np = get_heatmap(quadrant_image, fg_var, mask_var, fopa_model, device)
                mval = np.max(heatmap_np)
                if mval > best_value:
                    best_value = mval
                    best_variation = (fg_var, mask_var, np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape))
            end_var = time.time()
            print(f"Time for generating variations: {end_var - start_var:.2f}s")

            if best_variation is None:
                continue

            best_fg, best_mask, (ty, tx) = best_variation
            # Adjust to full‐image coordinates
            tx += quadrant_box[0]
            ty += quadrant_box[1]

            # 3) Blend FG with background patch (using alpha‐trimap blending)
            from matte_anything.helpers import blend_with_background
            blended_np = blend_with_background(np.array(best_fg), np.array(best_mask).astype(float) / 255.0, bg_path)

            start_blend = time.time()
            bg_image = overlay_images(bg_image, Image.fromarray(blended_np), best_mask, (tx, ty))
            end_blend = time.time()
            print(f"Time for blending/overlay: {end_blend - start_blend:.2f}s")

            plt.imshow(bg_image)
            plt.title(f"After adding to quadrant {quadrant_index + 1}")
            plt.axis("off")
            plt.show()

        # Save the final composite for this background
        final_path = os.path.join(output_dir, f"final_composite_{os.path.basename(bg_path)}")
        bg_image.save(final_path)
        plt.imshow(bg_image)
        plt.title("Final Composite Image")
        plt.axis("off")
        plt.show()

def generate_base_case_composites(
    background_dir: str,
    foreground_dir: str,
    output_dir: str,
    num_backgrounds: int = 100
):
    """
    Generate simple random‐rotation placements for up to `num_backgrounds` backgrounds.
    """
    os.makedirs(output_dir, exist_ok=True)
    background_files = [
        os.path.join(background_dir, f)
        for f in os.listdir(background_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:num_backgrounds]

    foreground_files = [
        os.path.join(foreground_dir, f)
        for f in os.listdir(foreground_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:4]

    for idx, bg_path in enumerate(background_files):
        bg_image = Image.open(bg_path).convert("RGBA")
        bg_w, bg_h = bg_image.size

        for fg_path in foreground_files:
            fg_image = Image.open(fg_path).convert("RGBA")
            fw, fh = fg_image.size
            scale_factor = min(bg_w / fw, bg_h / fh) * random.uniform(0.1, 0.3)
            new_size = (int(fw * scale_factor), int(fh * scale_factor))
            fg_resized = fg_image.resize(new_size, Image.LANCZOS)

            rot = random.randint(0, 360)
            fg_final = fg_resized.rotate(rot, expand=True)

            max_x = bg_w - fg_final.width
            max_y = bg_h - fg_final.height
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            bg_image.paste(fg_final, (x, y), fg_final)

        output_path = os.path.join(output_dir, f"base_case_composite_{idx + 1}.png")
        bg_image.save(output_path)
        print(f"Generated composite {idx + 1}/{len(background_files)}")
