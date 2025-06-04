import tensorflow as tf
import numpy as np
from PIL import Image
import os


def tensor_to_image(tensor):
    """
    Converts a single image tensor from [-1, 1] range to a standard [0, 255] PIL Image.
    Assumes input tensor has shape (H, W, 3).
    """
    # Undo normalization
    tensor = (tensor * 0.5) + 0.5
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)

    # Convert to uint8 numpy array
    image_array = (tensor * 255).numpy().astype(np.uint8)

    # PIL expects (H, W, C)
    return Image.fromarray(image_array)


def save_image_pair(input_tensor, output_tensor, save_dir, idx):
    """
    Saves an input/output image pair to disk for visual inspection.

    Parameters:
        input_tensor: Tensor of shape (1, H, W, 3)
        output_tensor: Tensor of shape (1, H, W, 3)
        save_dir: Base directory to save results under
        idx: Index number to differentiate filenames
    """
    # Convert tensors to images
    input_img = tensor_to_image(input_tensor[0])
    output_img = tensor_to_image(output_tensor[0])

    # Set up directories if they don't exist
    input_path = os.path.join(save_dir, "input")
    output_path = os.path.join(save_dir, "output")
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Save as PNGs with index
    input_img.save(os.path.join(input_path, f"input_{idx}.png"))
    output_img.save(os.path.join(output_path, f"output_{idx}.png"))
