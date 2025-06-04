import sys
import os
import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from segment_anything import sam_model_registry
GROUNDING_DINO_PATH = os.path.join(os.getcwd(), "GroundingDINO")

# Force GroundingDINO onto sys.path
if GROUNDING_DINO_PATH not in sys.path:
    sys.path.append(GROUNDING_DINO_PATH)

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model as dino_load_model

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_PATH = os.path.join(os.getcwd(), "Matte-Anything")

# Checkpoint paths
SAM_MODELS = {
    "vit_h": os.path.join(BASE_PATH, "pretrained/sam_vit_h_4b8939.pth"),
    "vit_b": os.path.join(BASE_PATH, "pretrained/sam_vit_b_01ec64.pth")
}

VITMATTE_MODELS = {
    "vit_b": os.path.join(BASE_PATH, "pretrained/ViTMatte_B_DIS.pth")
}

GROUNDING_DINO_CFG = {
    "weight": os.path.join(BASE_PATH, "pretrained/groundingdino_swint_ogc.pth"),
    "config": os.path.join(
        BASE_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
}

VITMATTE_CFG = {
    "vit_b": os.path.join(BASE_PATH, "configs/matte_anything.py")
}

def init_segment_anything(model_type: str):
    """
    Load a Segment Anything model (SAM) of type 'vit_h' or 'vit_b',
    return a SamPredictor object.
    """
    sam = sam_model_registry[model_type](checkpoint=SAM_MODELS[model_type]).to(device)
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)
    return predictor

def init_vitmatte(model_type: str):
    """
    Load a ViTMatte model (vit_b), return a ready-to-evaluate network.
    """
    cfg = LazyConfig.load(VITMATTE_CFG[model_type])
    vitmatte = instantiate(cfg.model)
    vitmatte.to(device)
    vitmatte.eval()
    DetectionCheckpointer(vitmatte).load(VITMATTE_MODELS[model_type])
    return vitmatte

def load_grounding_dino():
    """
    Load a GroundingDINO model from config+weights, return the model object.
    """
    cfg_path = GROUNDING_DINO_CFG["config"]
    weight_path = GROUNDING_DINO_CFG["weight"]
    model = dino_load_model(cfg_path, weight_path)
    return model

# Instantiate all three primary models here (so they are only created once)
SAM_PREDICTOR = init_segment_anything("vit_h")
VITMATTE_MODEL = init_vitmatte("vit_b")
GROUNDING_DINO_MODEL = load_grounding_dino()
