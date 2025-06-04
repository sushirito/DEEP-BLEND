import tensorflow as tf

# Scaling factor for cycle and identity losses
LAMBDA = 10.0

# Binary cross-entropy loss, we'll assume logits
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    """
    Calculates how well the discriminator separates real from fake.

    We want:
    - D(real) to be close to 1
    - D(fake) to be close to 0

    Returns:
        Average of real and fake loss, scaled down by 0.5
    """
    real_loss = loss_obj(tf.ones_like(real_output), real_output)
    fake_loss = loss_obj(tf.zeros_like(fake_output), fake_output)
    combined_loss = real_loss + fake_loss
    return 0.5 * combined_loss  # not dividing by 2 might be okay too, but sticking with paper


def generator_loss(fake_output):
    """
    How well the generator fools the discriminator.

    Ideally, D(G(x)) should be close to 1.

    Returns:
        Binary cross-entropy between predicted and target 1s
    """
    return loss_obj(tf.ones_like(fake_output), fake_output)


def cycle_consistency_loss(original_img, reconstructed_img):
    """
    Measures if translation preserves content through 
    A to B to A or B to A to B

    Scaled by lambda.
    """
    abs_diff = tf.abs(original_img - reconstructed_img)
    mean_abs_diff = tf.reduce_mean(abs_diff)
    return LAMBDA * mean_abs_diff  # could try L1 or L2


def identity_loss(input_img, same_output):
    """
    Encourages generators to not unnecessarily change images
    already in target domain.

    i.e. G(B) approx B

    Returns scaled mean absolute difference, halved per the paper.
    """
    l1_diff = tf.reduce_mean(tf.abs(input_img - same_output))
    return 0.5 * LAMBDA * l1_diff
