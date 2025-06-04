import os
import argparse
import subprocess
from PIL import Image
import numpy as np
import csv

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_images(folder_path, size=(299, 299)):
    """
    Load all images from the given folder, resize to target size (default 299x299),
    and return a numpy array of shape (N, H, W, C).
    """
    image_list = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        img_resized = img.resize(size, Image.BILINEAR)
        image_list.append(np.array(img_resized))

    if not image_list:
        raise ValueError(f"No images found in {folder_path}")

    return np.stack(image_list, axis=0)


def compute_fid(folderA, folderB):
    """
    Compute Frechet Inception Distance (FID) between two folders using `pytorch-fid`.

    This uses a subprocess call, so make sure `pytorch-fid` is in your PATH.
    """
    cmd = f"python -m pytorch_fid \"{folderA}\" \"{folderB}\" --dims 64"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()

    # Try to extract the last numeric value (usually the FID)
    try:
        fid = float(output.split()[-1])
    except Exception:
        raise RuntimeError(f"Couldn't parse FID output:\n{output}")

    return fid


def compute_ssim_psnr(imagesA, imagesB):
    """
    Compute SSIM and PSNR for each pair of images from two folders.
    Returns the average of both metrics across all pairs.
    """
    n = min(len(imagesA), len(imagesB))
    ssim_scores = []
    psnr_scores = []

    for i in range(n):
        imgA = imagesA[i]
        imgB = imagesB[i]
        ssim_score = ssim(imgA, imgB, channel_axis=2, data_range=255)
        psnr_score = psnr(imgA, imgB, data_range=255)
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)

    return float(np.mean(ssim_scores)), float(np.mean(psnr_scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folderA", required=True, help="First image folder (e.g., original inputs).")
    parser.add_argument("--folderB", required=True, help="Second image folder (e.g., model outputs).")
    parser.add_argument("--outdir", required=True, help="Folder to save results CSV.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading images from:\n  A: {args.folderA}\n  B: {args.folderB}")
    imgsA = load_images(args.folderA)
    imgsB = load_images(args.folderB)

    print("Computing FID...")
    fid_score = compute_fid(args.folderA, args.folderB)

    print("Computing SSIM and PSNR...")
    ssim_score, psnr_score = compute_ssim_psnr(imgsA, imgsB)

    # Save metrics to CSV
    output_csv = os.path.join(args.outdir, "fid_ssim_psnr.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["FID", fid_score])
        writer.writerow(["SSIM", ssim_score])
        writer.writerow(["PSNR", psnr_score])

    print(f"Results saved to {output_csv}")
    print(f"Summary:\n  FID: {fid_score:.4f}\n  SSIM: {ssim_score:.4f}\n  PSNR: {psnr_score:.4f}")


if __name__ == "__main__":
    main()