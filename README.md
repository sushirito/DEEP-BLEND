# DEEP-BLEND

## Overview
Macroplastics in waterways pose significant environmental and health risks, yet traditional deep-sea image data collection is prohibitively expensive, time-consuming, and geographically limited. To address these challenges, DEEP-BLEND offers a two-part pipeline for generating high-fidelity synthetic underwater pollution imagery. It integrates Realistic Multi-Object Composition (RMOC) to create lifelike composites of marine plastic debris and Simulating Underwater Lighting Effects (SULE)—an unpaired image-to-image translation network that adds realistic underwater lighting, color distortion, and blur. This synthetic dataset enables large-scale training and evaluation of pollution detection algorithms, bolsters environmental monitoring, and advances research on the impact of marine debris. The end-to-end system yields two key contributions:

1. **RMOC (Realistic Multi-Object Composition)**

   * Integrates state-of-the-art matting (SAM + ViTMatte + GroundingDINO) with FOPA (Fast Object Placement Assessment) to composite multiple terrestrial plastic objects seamlessly onto underwater backgrounds.
   * Produces anatomically and contextually plausible pollution scenes, addressing the scarcity of labeled underwater pollution data.

2. **SULE (Simulating Underwater Lighting Effects)**

   * Trains a CycleGAN on the LSUI dataset to learn unpaired mappings between terrestrial composites and authentic underwater imagery.
   * Applies realistic color attenuation, scattering, and blur to RMOC outputs, yielding final images that closely resemble true underwater photographs.

Together, RMOC + SULE generate diverse, labeled synthetic images that can be used to train object detectors, segmentation networks, and other downstream models for marine pollution research.


---

## Project Structure

```
DEEP-BLEND/                            
├── inputs/                        ← User-provided images
│   ├── backgrounds/               ← RMOC: marine background images
│   ├── foregrounds/               ← RMOC: terrestrial plastic images
│   └── cyclegan/                  
│       └── LSUI/
│           ├── GT/                ← SULE domain A (underwater images)
│           └── input/             ← SULE domain B (terrestrial images)
│
├── outputs/                       ← Generated results
│   ├── rmoc/                      ← RMOC composites
│   │   ├── composite/             ← FOPA-guided composites
│   │   └── random_baseline/       ← Randomly placed composites
│   │
│   ├── sule/                      ← SULE checkpoints & predictions
│   │   ├── train_outputs/         ← Previews & sample outputs per epoch
│   │   └── rmoc_to_underwater/    ← Inference on RMOC composites
│   │
│   └── metrics/                   ← Evaluation metric outputs (CSV files)
│       ├── fid_ssim_psnr/         
│       ├── niqe_brisque/          
│       └── clip_iqa/              
│
├── requirements.txt               ← All Python dependencies
├── setup.sh                       ← Shell script to install and clone repos
│
└── src/                           ← Source code
    ├── rmoc/                      ← RMOC (Realistic Multi-Object Composition)
    │   ├── matte_anything/        
    │   │   ├── model_init.py      ← SAM, ViTMatte, GroundingDINO loaders
    │   │   ├── helpers.py         ← Trimap, blending, pixel utilities
    │   │   └── inference.py       ← run_inference() → alpha mask + trimap
    │   │
    │   ├── fopa/
    │   │   ├── model_init.py      ← Load pretrained FOPA network
    │   │   ├── helpers.py         ← Image loading, variations, heatmap
    │   │   └── compositer.py      ← generate_composite_images() + base_case()
    │   │
    │   └── main.py                ← Entry point for RMOC pipelines
    │
    ├── sule/                      ← SULE (Simulating Underwater Lighting Effects)
    │   ├── data_pipeline.py       ← Load & preprocess LSUI dataset
    │   ├── model_factory.py       ← Build CycleGAN generators & discriminators
    │   ├── losses.py              ← Adversarial, cycle, identity losses
    │   ├── utils.py               ← Tensor↔PIL, checkpoint helpers
    │   ├── train.py               ← CycleGAN training loop & checkpointing
    │   ├── inference.py           ← Test-time inference & saving predictions
    │   └── main.py                ← CLI wrapper for training or inference
    │
    └── metrics/                   ← Evaluation metrics suite
        ├── fid_ssim_psnr.py       ← Compute FID, SSIM, PSNR between two folders
        ├── niqe_brisque.py        ← Compute NIQE & BRISQUE on paired folders
        └── clip_iqa.py            ← Compute CLIP-IQA prompt-based scores
```

---

## Usage Instructions

### 1. Installation & Setup

1. **Clone the repository** and enter its root directory:

   ```bash
   git clone https://github.com/YourUsername/MyRepo.git
   cd MyRepo
   chmod +x setup.sh
   ```

2. **Run the setup script** to install all dependencies (PyTorch, TensorFlow, CycleGAN, SAM, FOPA, metrics):

   ```bash
   ./setup.sh
   ```

   * This will clone Matte-Anything and GroundingDINO, download pretrained checkpoints, install RMOC requirements, install SULE (TensorFlow) requirements, and install metric libraries (`torchmetrics`, `pytorch-fid`, `scikit-image`, `brisque`, etc.).

3. **Verify dependencies** are installed. If any package fails, install manually:

   ```bash
   pip install -r requirements.txt
   ```

---

### 2. RMOC: Generating Realistic Composites

**Purpose**: Composite multiple plastic objects realistically onto a marine background.

1. **Place Inputs**:

   * Marine background images → `inputs/backgrounds/`
   * Terrestrial plastic images → `inputs/foregrounds/`

2. **Run RMOC**:

   ```bash
   python -m src.rmoc.main
   ```

   * **FOPA-guided composites** will be saved under:

     ```
     outputs/rmoc/composite/
     ```
   * **Random baseline composites** (random rotation & placement) under:

     ```
     outputs/rmoc/random_baseline/
     ```

3. **Inspect Outputs**:

   * View sample composites in `outputs/rmoc/composite/` to confirm realistic blending and placement.

---

### 3. SULE: CycleGAN-Based Underwater Simulation

**Purpose**: Transform terrestrial composites (or any input) into realistic underwater style.

#### 3.1 Dataset Preparation

1. **Download/add LSUI** unpaired dataset:

   * Underwater domain (GT) → `inputs/cyclegan/LSUI/GT/`
   * Terrestrial domain (input) → `inputs/cyclegan/LSUI/input/`

2. Verify that each folder contains `.png` or `.jpg` images (no subfolders).

#### 3.2 Training CycleGAN

```bash
python -m src.sule.main \
  --mode train \
  --data_A inputs/cyclegan/LSUI/GT \
  --data_B inputs/cyclegan/LSUI/input \
  --output outputs/sule/train_outputs
```

* **Arguments**:

  * `--mode train`
  * `--data_A` → path to underwater images (domain A)
  * `--data_B` → path to terrestrial images (domain B)
  * `--output` → directory for sample previews and logs

* **Outputs**:

  * **Checkpoints** under `checkpoints/train/` (one per 5 epochs, plus final).
  * **Sample previews** every 5 epochs under `outputs/sule/train_outputs/epoch_<n>/`.

#### 3.3 Inference (Apply to RMOC Composites)

```bash
python -m src.sule.main \
  --mode test \
  --weights checkpoints/train/final_generators/generator_g.h5 \
  --data_A outputs/rmoc/composite \
  --output outputs/sule/rmoc_to_underwater
```

* **Arguments**:

  * `--mode test`
  * `--weights` → path to `generator_g.h5` (trained G)
  * `--data_A` → folder of images to transform (e.g. RMOC composites)
  * `--output` → base folder for saving `input/` (original) and `output/` (underwater style)

* **Outputs**:

  * `outputs/sule/rmoc_to_underwater/input/` (original RMOC composite)
  * `outputs/sule/rmoc_to_underwater/output/` (underwater-styled composite)

---

### 4. Evaluation Metrics

All metric scripts accept two image folders (same number of files, sorted order) and save results to CSV.

#### 4.1 FID / SSIM / PSNR

```bash
python -m src.metrics.fid_ssim_psnr \
  --folderA <path_to_folderA> \
  --folderB <path_to_folderB> \
  --outdir outputs/metrics/fid_ssim_psnr
```

* **Example**:

  ```bash
  python -m src.metrics.fid_ssim_psnr \
    --folderA outputs/rmoc/composite \
    --folderB outputs/rmoc/random_baseline \
    --outdir outputs/metrics/fid_ssim_psnr
  ```
* **Output**:

  ```
  outputs/metrics/fid_ssim_psnr/fid_ssim_psnr.csv
  ```

  Contains:

  ```
  Metric,Value
  FID,6.79
  SSIM,0.8466
  PSNR,22.0557
  ```

#### 4.2 NIQE / BRISQUE

```bash
python -m src.metrics.niqe_brisque \
  --folder1 <path_to_folder1> \
  --folder2 <path_to_folder2> \
  --outdir outputs/metrics/niqe_brisque
```

* **Example**:

  ```bash
  python -m src.metrics.niqe_brisque \
    --folder1 outputs/rmoc/composite \
    --folder2 outputs/rmoc/random_baseline \
    --outdir outputs/metrics/niqe_brisque
  ```
* **Output**:

  ```
  outputs/metrics/niqe_brisque/niqe_brisque.csv
  ```

  Contains columns:

  ```
  Filename,NIQE_Folder1,NIQE_Folder2,BRISQUE_Folder1,BRISQUE_Folder2
  img001.png,5.0421,5.1773,25.2934,26.0732
  img002.png,3.9987,4.1342,20.1311,22.2205
  ...
  ```

#### 4.3 CLIP-IQA

```bash
python -m src.metrics.clip_iqa \
  --folder1 <path_to_folder1> \
  --folder2 <path_to_folder2> \
  --outdir outputs/metrics/clip_iqa
```

* **Example**:

  ```bash
  python -m src.metrics.clip_iqa \
    --folder1 outputs/rmoc/composite \
    --folder2 outputs/rmoc/random_baseline \
    --outdir outputs/metrics/clip_iqa
  ```
* **Outputs**:

  ```
  outputs/metrics/clip_iqa/clip_iqa_folder1.csv
  outputs/metrics/clip_iqa/clip_iqa_folder2.csv
  outputs/metrics/clip_iqa/clip_iqa_comparison.csv
  ```

  The comparison CSV merges side-by-side by filename.

---

## Directory Conventions

* **`inputs/`**: Place all user-provided images and datasets here.

  * RMOC: `inputs/backgrounds/`, `inputs/foregrounds/`
  * SULE: `inputs/cyclegan/LSUI/GT/`, `inputs/cyclegan/LSUI/input/`
* **`outputs/`**: All code-generated images and CSVs will be saved here.

  * RMOC composites: `outputs/rmoc/composite/`
  * RMOC baseline: `outputs/rmoc/random_baseline/`
  * SULE training previews: `outputs/sule/train_outputs/`
  * SULE inference: `outputs/sule/rmoc_to_underwater/`
  * Metrics:

    * `outputs/metrics/fid_ssim_psnr/`
    * `outputs/metrics/niqe_brisque/`
    * `outputs/metrics/clip_iqa/`
* **`checkpoints/train/`** (created automatically by SULE): Stores CycleGAN weights.

---

## Acknowledgements

This project builds on several foundational works and open-source contributions:

* **CycleGAN training and inference code** was adapted from the [TensorFlow CycleGAN Tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan), available at [`tensorflow/docs`](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb). We modified the code to suit the underwater translation needs of SULE.

* **FOPA (Fast Object Placement Assessment)** was implemented based on the original algorithm described by Niu et al.:

  ```bibtex
  @article{niu2022fast,
    title={Fast Object Placement Assessment},
    author={Niu, Li and Liu, Qingyang and Liu, Zhenchen and Li, Jiangtong},
    journal={arXiv preprint arXiv:2205.14280},
    year={2022}
  }
  ```

* **MatteAnything**, used for high-quality alpha matting, was integrated from the work of Yao et al.:

  ```bibtex
  @article{yao2024matte,
    title={Matte anything: Interactive natural image matting with segment anything model},
    author={Yao, Jingfeng and Wang, Xinggang and Ye, Lang and Liu, Wenyu},
    journal={Image and Vision Computing},
    pages={105067},
    year={2024},
    publisher={Elsevier}
  }
  ```

We thank the authors of these works for making their research and code available to the community. Additional attributions are provided in our paper.

---

## Citation

If you use DEEP-BLEND, please cite:

```bibtex
@inproceedings{shivakumar2024deep,
  title={DEEP-BLEND: Generating Adaptive Underwater Pollution Datasets},
  author={Shivakumar, Aditya},
  booktitle={2024 IEEE MIT Undergraduate Research Technology Conference (URTC)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
