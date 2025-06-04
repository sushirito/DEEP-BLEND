import os
import tensorflow as tf

# Some constants for image handling
AUTOTUNE = tf.data.AUTOTUNE
IMG_WIDTH, IMG_HEIGHT = 256, 256
BUFFER_SIZE = 1000
BATCH_SIZE = 1  # Might tweak this later for speed

def load_image_from_path(path):
    """
    Reads image from path and resizes it.
    Works with JPG or PNG, and ensures consistent 3-channel output.
    """
    byte_img = tf.io.read_file(path)
    img = tf.image.decode_image(byte_img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img


def random_crop(image):
    """
    Does a random crop and horizontal flip.
    We assume input is 286x286 at this point.
    """
    cropped = tf.image.random_crop(image, [IMG_HEIGHT, IMG_WIDTH, 3])
    flipped = tf.image.random_flip_left_right(cropped)
    return flipped


def normalize(image_tensor):
    """
    Normalize from [0,255] range to [-1, 1].
    Works for inference or training.
    """
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = (image_tensor / 127.5) - 1.0
    return image_tensor


def random_jitter(image):
    """
    Applies random resizing & cropping for training augmentation.
    Steps:
    1. Resize up to 286x286
    2. Crop to 256x256 randomly
    3. Flip randomly left-right
    """
    bigger = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    jittered = random_crop(bigger)
    return jittered


def preprocess_train(image_tensor):
    """
    Augment + normalize training image.
    """
    image_tensor = random_jitter(image_tensor)
    image_tensor = normalize(image_tensor)
    return image_tensor


def preprocess_test(image_tensor):
    """
    Just normalize test/val image (no augmentation).
    """
    return normalize(image_tensor)


def load_dataset_from_folder(folder_path, is_train=True):
    """
    Takes a folder of images and builds a tf.data.Dataset pipeline.

    - Reads image paths (PNG/JPG)
    - Loads and preprocesses
    - Shuffles, batches, prefetches if training
    """
    valid_extensions = (".png", ".jpg", ".jpeg")
    all_files = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith(valid_extensions)
    ]

    image_ds = tf.data.Dataset.from_tensor_slices(all_files)

    def _process_path(file_path):
        image = load_image_from_path(file_path)
        return preprocess_train(image) if is_train else preprocess_test(image)

    image_ds = image_ds.map(_process_path, num_parallel_calls=AUTOTUNE)

    if is_train:
        image_ds = image_ds.shuffle(BUFFER_SIZE)

    return image_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
