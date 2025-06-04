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

# Torch check without actually importing at top
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"


def generate_composite_images(background_dir, foreground_dir, output_dir):
    """
    Takes a set of background and foreground images,
    and creates composite images using MatteAnything + FOPA + quadrant blending.

    Shows intermediate results too.
    """
    os.makedirs(output_dir, exist_ok=True)

    bg_images = [
        os.path.join(background_dir, fname)
        for fname in os.listdir(background_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:100]

    fg_images = [
        os.path.join(foreground_dir, fname)
        for fname in os.listdir(foreground_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:4]

    fopa_model = load_fopa_model("best_weight.pth")

    for bg_path in bg_images:
        bg = Image.open(bg_path).convert("RGBA")
        quad_boxes = split_into_quadrants(bg)

        random.shuffle(fg_images)
        quad_ids = list(range(4))
        random.shuffle(quad_ids)

        for fg_path in fg_images:
            if not quad_ids:
                break

            q_id = quad_ids.pop()
            q_box = quad_boxes[q_id]
            q_crop = bg.crop(q_box)

            # MatteAnything mask extraction
            from matte_anything.inference import run_inference
            t1 = time.time()
            alpha, _, trimap = run_inference(fg_path)
            t2 = time.time()
            print(f"Inference took {t2 - t1:.2f}s")

            mask = (alpha * 255).astype(np.uint8)
            grayscale = Image.fromarray(mask)
            mask_name = os.path.splitext(os.path.basename(fg_path))[0] + "_mask.png"
            grayscale.save(os.path.join(output_dir, mask_name))

            # Generate variations and pick using heatmap
            variants = generate_variations(fg_path, os.path.join(output_dir, mask_name))
            best = None
            best_score = -np.inf

            t3 = time.time()
            for fg_var, mask_var in variants:
                heatmap = get_heatmap(q_crop, fg_var, mask_var, fopa_model, device)
                score = np.max(heatmap)
                if score > best_score:
                    best_score = score
                    best = (fg_var, mask_var, np.unravel_index(np.argmax(heatmap), heatmap.shape))
            t4 = time.time()
            print(f"Variation scoring took {t4 - t3:.2f}s")

            if best is None:
                print("No good variation found — skipping.")
                continue

            best_fg, best_mask, (ty, tx) = best
            tx += q_box[0]
            ty += q_box[1]

            # Blending
            from matte_anything.helpers import blend_with_background
            blended_array = blend_with_background(np.array(best_fg), np.array(best_mask).astype(float) / 255.0, bg_path)

            t5 = time.time()
            bg = overlay_images(bg, Image.fromarray(blended_array), best_mask, (tx, ty))
            t6 = time.time()
            print(f"Overlay done in {t6 - t5:.2f}s")

            plt.imshow(bg)
            plt.title(f"Added to quadrant {q_id + 1}")
            plt.axis("off")
            plt.show()

        final_name = os.path.join(output_dir, f"final_composite_{os.path.basename(bg_path)}")
        bg.save(final_name)
        plt.imshow(bg)
        plt.title("Final Composite")
        plt.axis("off")
        plt.show()


def generate_base_case_composites(bg_dir, fg_dir, out_dir, num_backgrounds=100):
    """
    Simpler baseline generator — randomly pastes resized & rotated FG images on BGs.
    No FOPA/MatteAnything involved.
    """
    os.makedirs(out_dir, exist_ok=True)

    bg_files = [
        os.path.join(bg_dir, f)
        for f in os.listdir(bg_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:num_backgrounds]

    fg_files = [
        os.path.join(fg_dir, f)
        for f in os.listdir(fg_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:4]

    for idx, bg_path in enumerate(bg_files):
        bg = Image.open(bg_path).convert("RGBA")
        bg_w, bg_h = bg.size

        for fg_path in fg_files:
            fg = Image.open(fg_path).convert("RGBA")
            fg_w, fg_h = fg.size
            scale = min(bg_w / fg_w, bg_h / fg_h) * random.uniform(0.1, 0.3)
            resized = fg.resize((int(fg_w * scale), int(fg_h * scale)), Image.LANCZOS)

            rotated = resized.rotate(random.randint(0, 360), expand=True)

            max_x = bg_w - rotated.width
            max_y = bg_h - rotated.height
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            bg.paste(rotated, (paste_x, paste_y), rotated)

        save_path = os.path.join(out_dir, f"base_case_composite_{idx + 1}.png")
        bg.save(save_path)
        print(f"Saved base composite: {save_path}")
