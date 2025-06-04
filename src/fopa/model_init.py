import sys
import os
import torch

# Make sure backbone/ and network/ (from FOPA repo) are on sys.path
ROOT = os.getcwd()
for sub in ["network", "backbone"]:
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.append(path)

# Import the ResNet‐based FOPA model
from backbone.ResNet import ResNet, Backbone_ResNet18_in3, Backbone_ResNet18_in3_1
from network.ObPlaNet_simple import ObPlaNet_resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_fopa_model(model_path: str = "best_weight.pth"):
    """
    Instantiate ObPlaNet_resnet18, load weights, set to eval, and return model.
    """
    model = ObPlaNet_resnet18()
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
