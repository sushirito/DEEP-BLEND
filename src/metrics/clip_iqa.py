import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchmetrics.functional.multimodal import clip_image_quality_assessment

# Define labels for each prompt pair (these will be CSV columns)
CUSTOM_PROMPTS = (
    "AccurateHues",
    "ProperAttenuation",
    "RealisticScatterBlur",
    "NaturalCaustics"
)

# Each pair: (good description, bad description)
PROMPT_PAIRS = (
    ("Accurate blue-green hues typical of underwater scenes.", "Inaccurate color hues not resembling underwater scenes."),
    ("Proper light attenuation mimicking underwater depth.", "Improper light attenuation failing to mimic underwater depth."),
    ("Realistic scatter and blur effects on distant objects.", "Unrealistic scatter and blur effects on distant objects."),
    ("Natural caustic light patterns on surfaces.", "Unnatural caustic light patterns on surfaces.")
)

def get_region(index):
    """
    Assign each image a region name based on its index.
    This assumes a fixed order in the folder (hardcoded for our application, but good enough here).
    """
    if index < 60:
        return "Monterey"
    elif index < 90:
        return "Lake Tahoe"
    else:
        return "Lexington Reservoir"

def process_folder(folder_path):
    """
    Evaluate each image in the folder using CLIP-IQA against all prompt pairs.
    Returns a list of dictionaries with per-image scores.
    """
    results = []

    for idx, fname in enumerate(sorted(os.listdir(folder_path))):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path).convert("RGB")

        # Convert to tensor format expected by torchmetrics
        arr = np.asarray(img)
        tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float()

        # Run CLIP-IQA with our custom prompts
        scores = clip_image_quality_assessment(
            tensor,
            model_name_or_path="clip_iqa",
            data_range=255,
            prompts=PROMPT_PAIRS
        )

        # Format results for this image
        result = {"filename": fname, "region": get_region(idx)}
        for label, score in zip(CUSTOM_PROMPTS, scores.values()):
            result[label] = float(score.item())

        results.append(result)

    return results

def summarize_and_save(folder1, folder2, outdir):
    """
    Runs CLIP-IQA on two folders, saves separate CSVs, and combines into one comparison table.
    """
    os.makedirs(outdir, exist_ok=True)

    # First folder
    data1 = process_folder(folder1)
    df1 = pd.DataFrame(data1)
    path1 = os.path.join(outdir, "clip_iqa_folder1.csv")
    df1.to_csv(path1, index=False)
    print("Saved:", path1)

    # Second folder
    data2 = process_folder(folder2)
    df2 = pd.DataFrame(data2)
    path2 = os.path.join(outdir, "clip_iqa_folder2.csv")
    df2.to_csv(path2, index=False)
    print("Saved:", path2)

    # Combine results side-by-side using filename + region
    merged = pd.merge(df1, df2, on=["filename", "region"], suffixes=("_A", "_B"))
    combined_path = os.path.join(outdir, "clip_iqa_comparison.csv")
    merged.to_csv(combined_path, index=False)
    print("Saved:", combined_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", required=True, help="Path to first image folder.")
    parser.add_argument("--folder2", required=True, help="Path to second image folder.")
    parser.add_argument("--outdir", required=True, help="Output directory for CSV files.")
    args = parser.parse_args()

    summarize_and_save(args.folder1, args.folder2, args.outdir)

if __name__ == "__main__":
    main()
