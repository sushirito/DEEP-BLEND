import os
import torch
import numpy as np
from PIL import Image
from torchvision.ops import box_convert

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

from .model_init import SAM_PREDICTOR as predictor
from .model_init import VITMATTE_MODEL as vitmatte
from .model_init import GROUNDING_DINO_MODEL as grounding_dino

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import predict as dino_predict

from .helpers import generate_trimap, convert_pixels

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_inference(image_path):
    """
    Full matte generation pipeline:
    - Detects foreground with GroundingDINO
    - Creates trimap using SAM
    - Refines trimap with secondary object detection
    - Runs ViTMatte to estimate alpha matte

    Returns:
        alpha (float32 np.ndarray): matte prediction in [0,1]
        input_x (uint8 np.ndarray): original RGB image
        trimap (float32 np.ndarray): intermediate mask values in {0.0, 0.5, 1.0}
    """
    input_image = Image.open(image_path).convert("RGB")
    input_x = np.array(input_image)  # raw RGB data

    # Prep SAM predictor
    predictor.set_image(input_x)

    # Get object boxes from GroundingDINO using generic caption
    dino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformed_img, _ = dino_transform(input_image, None)

    fg_caption = "foreground object"
    fg_box_thresh = 0.25
    fg_text_thresh = 0.25

    fg_boxes, _, _ = dino_predict(
        model=grounding_dino,
        image=transformed_img,
        caption=fg_caption,
        box_threshold=fg_box_thresh,
        text_threshold=fg_text_thresh,
        device=device
    )

    # Convert box format and scale to match original image size
    if fg_boxes.shape[0] > 0:
        h, w, _ = input_x.shape
        fg_boxes = fg_boxes * torch.tensor([w, h, w, h]).to(device)
        fg_boxes = box_convert(fg_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        fg_boxes = predictor.transform.apply_boxes_torch(fg_boxes, input_x.shape[:2])
    else:
        fg_boxes = None

    # Predict initial binary mask using SAM
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=fg_boxes,
        multimask_output=False
    )
    mask_np = (masks[0][0].cpu().numpy().astype(np.uint8)) * 255

    # Generate trimap from the binary mask
    trimap = generate_trimap(mask_np).astype(np.float32)
    trimap[trimap == 128] = 0.5
    trimap[trimap == 255] = 1.0

    # Run secondary DINO check for trash-type objects
    trash_caption = "plastic material, trash bag, straw, spoon, bag, container"
    trash_boxes, _, _ = dino_predict(
        model=grounding_dino,
        image=transformed_img,
        caption=trash_caption,
        box_threshold=0.5,
        text_threshold=0.25,
        device=device
    )

    if trash_boxes.shape[0] > 0:
        boxes_scaled = trash_boxes * torch.tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        trimap = convert_pixels(trimap, boxes_xyxy)

    # Final alpha matte prediction using ViTMatte
    input_data = {
        "image": torch.from_numpy(input_x).permute(2, 0, 1).unsqueeze(0) / 255.0,
        "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0)
    }

    with torch.no_grad():
        alpha_tensor = vitmatte(input_data)["phas"]
    alpha = alpha_tensor.flatten(0, 2).cpu().numpy()

    # Optional: create visual output if needed
    rgba = Image.fromarray(input_x).convert("RGBA")
    alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
    rgba.putalpha(alpha_img)

    return alpha, input_x, trimap
