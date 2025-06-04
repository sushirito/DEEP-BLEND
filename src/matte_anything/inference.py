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


def run_inference(image_path: str):
    """
    Perform the MatteAnything inference pipeline on a single image.

    Returns:
      alpha   : float32 array of shape (H, W) representing the predicted alpha matte
      input_x : uint8 array of shape (H, W, 3) containing the original RGB image
      trimap  : float32 array of shape (H, W) with values in {0.0, 0.5, 1.0}
    """
    input_image = Image.open(image_path).convert("RGB")
    input_x = np.array(input_image)

    predictor.set_image(input_x)

    # Prep the image for GroundingDINO and run object detection for a generic "foreground object"
    dino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = dino_transform(input_image, None)

    fg_caption = "foreground object"
    fg_box_threshold = 0.25
    fg_text_threshold = 0.25

    fg_boxes, logits, phrases = dino_predict(
        model=grounding_dino,
        image=image_transformed,
        caption=fg_caption,
        box_threshold=fg_box_threshold,
        text_threshold=fg_text_threshold,
        device=device
    )

    if fg_boxes.shape[0] == 0:
        transformed_boxes = None
    else:
        h, w, _ = input_x.shape
        fg_boxes = torch.Tensor(fg_boxes).to(device)
        # Scale the box coordinates to the original image size
        fg_boxes = fg_boxes * torch.Tensor([w, h, w, h]).to(device)
        # Convert from center-based format (cx, cy, w, h) to (x_min, y_min, x_max, y_max)
        fg_boxes = box_convert(boxes=fg_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        transformed_boxes = predictor.transform.apply_boxes_torch(
            fg_boxes, input_x.shape[:2]
        )

    # Use SAM to predict a mask for the detected box
    masks, scores, logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )
    masks = masks.cpu().detach().numpy()
    mask = (masks[0][0].astype(np.uint8)) * 255

    # Create a trimap from the binary mask
    trimap = generate_trimap(mask).astype(np.float32)
    trimap[trimap == 128] = 0.5
    trimap[trimap == 255] = 1.0

    # Run 2nd GroundingDINO pass for trash-related items, update the trimap
    tr_caption = "plastic material, trash bag, straw, spoon, bag, container"
    tr_box_threshold = 0.5
    tr_text_threshold = 0.25

    boxes, logits, phrases = dino_predict(
        model=grounding_dino,
        image=image_transformed,
        caption=tr_caption,
        box_threshold=tr_box_threshold,
        text_threshold=tr_text_threshold,
        device=device
    )

    if boxes.shape[0] != 0:
        h, w, _ = input_x.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        trimap = convert_pixels(trimap, xyxy)

    # Run ViTMatte to obtain the alpha matte
    input_data = {
        "image": torch.from_numpy(input_x).permute(2, 0, 1).unsqueeze(0) / 255.0,
        "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
    }
    alpha_tensor = vitmatte(input_data)["phas"].flatten(0, 2).detach().cpu().numpy()

    # Create an RGBA image combining the original RGB and the alpha channel
    rgba_image = Image.fromarray(input_x).convert("RGBA")
    alpha_channel = (alpha_tensor * 255).astype(np.uint8)
    rgba_image.putalpha(Image.fromarray(alpha_channel))

    return alpha_tensor, input_x, trimap
