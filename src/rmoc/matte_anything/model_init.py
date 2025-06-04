import sys
import os
import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from segment_anything import sam_model_registry

# Force GroundingDINO repo into path if it's not already
GROUNDING_DINO_PATH = os.path.join(os.getcwd(), "GroundingDINO")
if GROUNDING_DINO_PATH not in sys.path:
    sys.path.append(GROUNDING_DINO_PATH)

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model as dino_load_model

# Default to GPU if it's available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Base directory for pretrained weights and configs
BASE_PATH = os.path.join(os.getcwd(), "Matte-Anything")

# Checkpoint locations
SAM_MODELS = {
    "vit_h": os.path.join(BASE_PATH, "pretrained/sam_vit_h_4b8939.pth"),
    "vit_b": os.path.join(BASE_PATH, "pretrained/sam_vit_b_01ec64.pth")
}

VITMATTE_MODELS = {
    "vit_b": os.path.join(BASE_PATH, "pretrained/ViTMatte_B_DIS.pth")
}

GROUNDING_DINO_CFG = {
    "weight": os.path.join(BASE_PATH, "pretrained/groundingdino_swint_ogc.pth"),
    "config": os.path.join(BASE_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
}

VITMATTE_CFG = {
    "vit_b": os.path.join(BASE_PATH, "configs/matte_anything.py")
}


def init_segment_anything(model_type):
    """
    Loads a SAM model and wraps it in a SamPredictor.
    model_type should be either 'vit_h' or 'vit_b'.
    """
    checkpoint = SAM_MODELS[model_type]
    model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

    from segment_anything import SamPredictor
    return SamPredictor(model)


def init_vitmatte(model_type):
    """
    Loads ViTMatte model from config + weights.
    Returns a ready-to-use PyTorch model in eval mode.
    """
    config_path = VITMATTE_CFG[model_type]
    model_weights = VITMATTE_MODELS[model_type]

    cfg = LazyConfig.load(config_path)
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()

    DetectionCheckpointer(model).load(model_weights)
    return model


def load_grounding_dino():
    """
    Loads GroundingDINO from provided Swin config and pretrained weights.
    """
    cfg_file = GROUNDING_DINO_CFG["config"]
    weight_file = GROUNDING_DINO_CFG["weight"]
    return dino_load_model(cfg_file, weight_file)


# Eagerly load all three major components once
SAM_PREDICTOR = init_segment_anything("vit_h")
VITMATTE_MODEL = init_vitmatte("vit_b")
GROUNDING_DINO_MODEL = load_grounding_dino()