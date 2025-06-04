import sys
import os
import torch

# Add 'network' and 'backbone' subfolders to sys.path manually (from FOPA repo)
# So we can import the custom model components
ROOT_DIR = os.getcwd()
for subfolder in ["network", "backbone"]:
    full_path = os.path.join(ROOT_DIR, subfolder)
    if full_path not in sys.path:
        sys.path.append(full_path)

# Now safe to import model definitions
from backbone.ResNet import ResNet, Backbone_ResNet18_in3, Backbone_ResNet18_in3_1
from network.ObPlaNet_simple import ObPlaNet_resnet18

# Default to CUDA if it's available
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_fopa_model(model_path="best_weight.pth"):
    """
    Loads the pre-trained FOPA model (ObPlaNet with ResNet-18 backbone).

    Steps:
    - Initializes architecture
    - Loads saved weights
    - Sends model to GPU/CPU
    - Sets model to eval mode

    Returns:
        The ready-to-use PyTorch model.
    """
    model = ObPlaNet_resnet18()
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model
