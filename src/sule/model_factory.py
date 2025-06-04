import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

# Generator and discriminator will both work with RGB images
OUTPUT_CHANNELS = 3


def build_generator():
    """
    Constructs a U-Net style generator using InstanceNorm.
    Used for both G: A → B and F: B → A mappings.

    Output shape: (batch, 256, 256, 3)
    """
    gen = pix2pix.unet_generator(
        OUTPUT_CHANNELS, norm_type='instancenorm'
    )
    return gen  # might want to print summary during dev


def build_discriminator():
    """
    Builds a PatchGAN-style discriminator.
    Used for distinguishing real vs generated images.

    Note: outputs 30x30 patch map instead of a single scalar.
    """
    disc = pix2pix.discriminator(
        norm_type='instancenorm', target=False  # no target input in CycleGAN
    )
    return disc
