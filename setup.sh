# Clone + prep Matte-Anything
git clone https://github.com/hustvl/Matte-Anything.git
cd Matte-Anything
mkdir -p ./pretrained

# Download pretrained weights
wget -O ./pretrained/sam_vit_h_4b8939.pth \
     https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth
wget -O ./pretrained/sam_vit_b_01ec64.pth \
     https://huggingface.co/spaces/jbrinkma/segment-anything/resolve/main/sam_vit_b_01ec64.pth
wget -O ./pretrained/ViTMatte_B_DIS.pth \
     https://huggingface.co/nielsr/vitmatte-checkpoints/resolve/main/ViTMatte_B_DIS.pth
wget -O ./pretrained/groundingdino_swint_ogc.pth \
     https://huggingface.co/alexgenovese/background-workflow/resolve/1cbf8c24aa8a2e8d5ca6871800442b35ff6f9d48/groundingdino_swint_ogc.pth

# Python dependencies for MatteAnything
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -r requirements.txt

# Clone & install GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git GroundingDINO
cd GroundingDINO
pip install -e .
pip install fairscale
cd ..

# 2. Clone/prep FOPA
git clone https://github.com/bcmi/FOPA-Fast-Object-Placement-Assessment.git fopa
cd fopa
pip install -r requirements.txt

# Download FOPA weights and data
gdown --id 1HTP6bSmuMb2Dux3vEX2fJc3apjLBjy0q -O best_weight.pth
gdown --id 1VBTCO3QT1hqzXre1wdWlndJR97SI650d -O data.zip
gdown --id 1DMCINPzrBsxXj_9fTKnzB7mQcd8WQi3T -O SOPA.pth.tar

unzip data.zip -d data/data
mv SOPA.pth.tar data/data/
mkdir -p data/data

cd ..
# Move everything from fopa/* into project root (so that backbone/ and network/ are on PYTHONPATH)
mv fopa/* .