import os
import argparse
import csv
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage.filters import convolve
from scipy.special import gamma
from tqdm import tqdm

NIQE_PARAMS_URL = "https://raw.githubusercontent.com/xinntao/BasicSR/master/basicsr/metrics/niqe_pris_params.npz"
NIQE_PARAMS_LOCAL = "niqe_pris_params.npz"

def ensure_niqe_params():
    """Download the NIQE parameter file if it's not already present."""
    if not os.path.exists(NIQE_PARAMS_LOCAL):
        print("Downloading NIQE parameter file...")
        os.system(f"wget {NIQE_PARAMS_URL} -O {NIQE_PARAMS_LOCAL}")

mu_pris_param = None
cov_pris_param = None
gaussian_window = None

def load_niqe_config():
    """Load global NIQE parameters from local .npz file."""
    global mu_pris_param, cov_pris_param, gaussian_window
    data = np.load(NIQE_PARAMS_LOCAL)
    mu_pris_param = data["mu_pris_param"]
    cov_pris_param = data["cov_pris_param"]
    gaussian_window = data["gaussian_window"]

def estimate_aggd_params(patch):
    """Estimate AGGD params (a, bl, br) for a given image patch."""
    gam_range = np.arange(0.2, 10.001, 0.001)
    gam_recip = 1.0 / gam_range
    r_gam = (np.square(gamma(gam_recip * 2)) /
             (gamma(gam_recip) * gamma(gam_recip * 3)))

    patch_flat = patch.flatten()
    left_std = np.sqrt(np.mean(patch_flat[patch_flat < 0]**2))
    right_std = np.sqrt(np.mean(patch_flat[patch_flat > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(patch_flat)))**2 / np.mean(patch_flat**2)
    rhatnorm = rhat * (gammahat**3 + 1) * (gammahat + 1) / ((gammahat**2 + 1)**2)

    idx = np.argmin((r_gam - rhatnorm)**2)
    alpha = gam_range[idx]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))

    return alpha, beta_l, beta_r

def extract_patch_features(patch):
    """Get NIQE features from one image patch."""
    features = []
    alpha, bl, br = estimate_aggd_params(patch)
    features.extend([alpha, (bl + br) / 2])

    for dx, dy in [(0,1), (1,0), (1,1), (1,-1)]:
        shifted = np.roll(patch, shift=(dx, dy), axis=(0, 1))
        alpha_s, bl_s, br_s = estimate_aggd_params(patch * shifted)
        mean_shift = (br_s - bl_s) * (gamma(2 / alpha_s) / gamma(1 / alpha_s))
        features.extend([alpha_s, mean_shift, bl_s, br_s])
    return features

def compute_niqe(image):
    """Run full NIQE pipeline on a grayscale image."""
    h, w = image.shape
    h_blocks, w_blocks = h // 96, w // 96
    cropped = image[:h_blocks * 96, :w_blocks * 96]

    all_features = []
    for scale in (1, 2):
        mu = convolve(cropped, gaussian_window, mode="nearest")
        sigma = np.sqrt(np.abs(convolve(cropped**2, gaussian_window, mode="nearest") - mu**2))
        normalized = (cropped - mu) / (sigma + 1)

        features = []
        for i in range(h_blocks):
            for j in range(w_blocks):
                patch = normalized[i * 96:(i+1) * 96, j * 96:(j+1) * 96]
                features.append(extract_patch_features(patch))
        all_features.append(np.stack(features, axis=0))

        if scale == 1:
            small = cv2.resize(cropped / 255.0, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR) * 255
            cropped = small.astype(np.float32)

    all_features = np.concatenate(all_features, axis=1)
    mu_dist = np.nanmean(all_features, axis=0)
    valid = all_features[~np.isnan(all_features).any(axis=1)]
    cov_dist = np.cov(valid, rowvar=False)

    invcov = np.linalg.pinv((cov_pris_param + cov_dist) / 2)
    diff = mu_pris_param - mu_dist
    score = np.sqrt(diff @ invcov @ diff.T)
    return score

def evaluate_niqe(folder):
    """Return NIQE scores for each image in a folder."""
    scores = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            continue
        img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        scores.append(compute_niqe(gray.astype(np.float32)))
    return scores

def evaluate_brisque(folder):
    """Return BRISQUE scores for each image in a folder."""
    from brisque import BRISQUE
    scorer = BRISQUE(url=False)
    results = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            arr = np.asarray(img)
            score = scorer.score(img=arr)
            results.append((fname, score))
    return results

def save_results(folder1, folder2, outdir):
    ensure_niqe_params()
    load_niqe_config()

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "niqe_brisque.csv")

    scores1 = evaluate_niqe(folder1)
    scores2 = evaluate_niqe(folder2)
    brisque1 = evaluate_brisque(folder1)
    brisque2 = evaluate_brisque(folder2)

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "NIQE_1", "NIQE_2", "BRISQUE_1", "BRISQUE_2"])
        for i in range(min(len(scores1), len(scores2), len(brisque1), len(brisque2))):
            fn1 = os.path.basename(brisque1[i][0])
            fn2 = os.path.basename(brisque2[i][0])
            assert fn1 == fn2, f"Mismatch in file names: {fn1} vs {fn2}"
            writer.writerow([
                fn1,
                f"{scores1[i]:.4f}", f"{scores2[i]:.4f}",
                f"{brisque1[i][1]:.4f}", f"{brisque2[i][1]:.4f}"
            ])

    print(f"Results written to {outfile}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", required=True, help="Path to first folder of images.")
    parser.add_argument("--folder2", required=True, help="Path to second folder of images.")
    parser.add_argument("--outdir", required=True, help="Folder to save evaluation CSV.")
    args = parser.parse_args()
    save_results(args.folder1, args.folder2, args.outdir)

if __name__ == "__main__":
    main()