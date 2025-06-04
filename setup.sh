#!/usr/bin/env bash

########################################
# 1. Setup: Dependencies for CycleGAN (SULE)
########################################
echo "Installing core CycleGAN (TensorFlow) dependencies..."
pip install "tensorflow>=2.8.0"
pip install tensorflow-datasets
pip install git+https://github.com/tensorflow/examples.git
pip install scikit-learn pandas

########################################
# 2. Install Metric Libraries for Evaluation
########################################
echo "Installing metric libraries (torchmetrics, FID, BRISQUE, etc)..."
pip install torchmetrics[image] piq pytorch-fid
pip install scikit-image brisque opencv-python-headless scipy tqdm

########################################
# 3. Clone and Prepare Matte-Anything (RMOC)
########################################
echo "Cloning Matte-Anything repository and downloading pretrained weights..."
git clone https://github.com/hustvl/Matte-Anything.git
cd Matte-Anything
mkdir -p ./pretrained

# Download pretrained model weights
wget -O ./pretrained/sam_vit_h_4b8939.pth \
     https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth
wget -O ./pretrained/sam_vit_b_01ec64.pth \
     https://huggingface.co/spaces/jbrinkma/segment-anything/resolve/main/sam_vit_b_01ec64.pth
wget -O ./pretrained/ViTMatte_B_DIS.pth \
     https://huggingface.co/nielsr/vitmatte-checkpoints/resolve/main/ViTMatte_B_DIS.pth
wget -O ./pretrained/groundingdino_swint_ogc.pth \
     https://huggingface.co/alexgenovese/background-workflow/resolve/1cbf8c24aa8a2e8d5ca6871800442b35ff6f9d48/groundingdino_swint_ogc.pth

# Install SAM + Detectron2
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install any other MatteAnything-specific dependencies
pip install -r requirements.txt

# Clone and install GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git GroundingDINO
cd GroundingDINO
pip install -e .
pip install fairscale
cd ..

########################################
# 4. Clone and Setup FOPA (RMOC Placement)
########################################
echo "Cloning and preparing FOPA placement model..."
git clone https://github.com/bcmi/FOPA-Fast-Object-Placement-Assessment.git fopa
cd fopa
pip install -r requirements.txt

# Download pretrained weights and test data
gdown --id 1HTP6bSmuMb2Dux3vEX2fJc3apjLBjy0q -O best_weight.pth
gdown --id 1VBTCO3QT1hqzXre1wdWlndJR97SI650d -O data.zip
gdown --id 1DMCINPzrBsxXj_9fTKnzB7mQcd8WQi3T -O SOPA.pth.tar

# Unpack and move assets into expected layout
unzip data.zip -d data/data
mv SOPA.pth.tar data/data/
mkdir -p data/data  # Safe-guard (redundant but defensive)

cd ..
mv fopa/* .

echo "Setup complete. You're ready to run inference or training!"