import os
import glob
import tensorflow as tf
from PIL import Image
import numpy as np

# Local modules
from .data_pipeline import preprocess_test
from .model_factory import build_generator
from .utils import tensor_to_image


def load_weights_into_generator(weights_file):
    """
    Initializes a generator model and loads weights into it.
    Make sure the architecture matches what was used during training!
    """
    gen_model = build_generator()
    gen_model.load_weights(weights_file)
    return gen_model


def preprocess_single_image(image_path):
    """
    Loads an image from disk and prepares it for inference:
    - Resize to 256x256
    - Normalize to [-1, 1]
    - Add batch dimension
    """
    raw_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(raw_bytes, channels=3)
    image = tf.image.resize(image, (256, 256))  # hardcoding this for now
    image = preprocess_test(image)  # normalize etc.
    return tf.expand_dims(image, axis=0)  # shape becomes (1, H, W, C)


def save_test_predictions(generator_model, input_dir, output_dir):
    """
    Takes a folder of input images, passes them through the generator,
    and saves both input and predicted images in the output folder.
    """
    # Find all images in supported formats
    image_files = []
    image_files += glob.glob(os.path.join(input_dir, "*.png"))
    image_files += glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files += glob.glob(os.path.join(input_dir, "*.jpeg"))

    os.makedirs(output_dir, exist_ok=True)

    for idx, img_path in enumerate(image_files):
        # Load and preprocess
        input_tensor = preprocess_single_image(img_path)
        output_tensor = generator_model(input_tensor, training=False)

        # Convert back to PIL images
        input_image = tensor_to_image(input_tensor[0])
        output_image = tensor_to_image(output_tensor[0])

        # Save paths
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        input_save_dir = os.path.join(output_dir, "input")
        output_save_dir = os.path.join(output_dir, "output")

        # Ensure dirs exist
        os.makedirs(input_save_dir, exist_ok=True)
        os.makedirs(output_save_dir, exist_ok=True)

        input_image.save(os.path.join(input_save_dir, f"{base_name}.png"))
        output_image.save(os.path.join(output_save_dir, f"{base_name}_pred.png"))

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} of {len(image_files)} images...")

    print(f"Done! Saved results to: {output_dir}")