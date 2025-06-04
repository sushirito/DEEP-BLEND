import os
import time
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import Adam

from .data_pipeline import load_dataset_from_folder
from .model_factory import build_generator, build_discriminator
from .losses import (
    generator_loss, discriminator_loss,
    cycle_consistency_loss, identity_loss
)
from .utils import save_image_pair

# --- Config (could make this a CLI arg system later maybe)
EPOCHS = 20
CHECKPOINT_DIR = "checkpoints/train"
LOSS_LOG_PATH = "losses.csv"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

def train_cycle_gan(train_A_path, train_B_path, preview_dir_root):
    """
    Starts training for CycleGAN using folders A and B.
    A = original images, B = target images
    """
    # Initialize our models
    G_A2B = build_generator()     # A -> B direction
    G_B2A = build_generator()     # B -> A direction
    D_A = build_discriminator()   # Discriminator for A
    D_B = build_discriminator()   # Discriminator for B

    # Using same optimizer config across all
    optimizer_GA = Adam(2e-4, beta_1=0.5)
    optimizer_GB = Adam(2e-4, beta_1=0.5)
    optimizer_DA = Adam(2e-4, beta_1=0.5)
    optimizer_DB = Adam(2e-4, beta_1=0.5)

    # Load checkpoint if available
    checkpoint = tf.train.Checkpoint(
        G_A2B=G_A2B, G_B2A=G_B2A,
        D_A=D_A, D_B=D_B,
        optimizer_GA=optimizer_GA,
        optimizer_GB=optimizer_GB,
        optimizer_DA=optimizer_DA,
        optimizer_DB=optimizer_DB
    )
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored checkpoint from {manager.latest_checkpoint}")

    # Load datasets from folders
    dataset_A = load_dataset_from_folder(train_A_path, is_train=True)
    dataset_B = load_dataset_from_folder(train_B_path, is_train=True)

    # Keep track of all loss values over time
    all_losses = pd.DataFrame(columns=[
        "epoch", "gen_g_loss", "gen_f_loss",
        "cycle_loss_A", "cycle_loss_B",
        "disc_x_loss", "disc_y_loss",
        "identity_loss_A", "identity_loss_B"
    ])

    # --- Inner train function ---
    @tf.function
    def train_single_batch(real_A, real_B):
        with tf.GradientTape(persistent=True) as tape:
            # Forward cycles
            fake_B = G_A2B(real_A, training=True)
            recov_A = G_B2A(fake_B, training=True)

            fake_A = G_B2A(real_B, training=True)
            recov_B = G_A2B(fake_A, training=True)

            # Identity maps
            same_A = G_B2A(real_A, training=True)
            same_B = G_A2B(real_B, training=True)

            # Discriminator predictions
            D_real_A = D_A(real_A, training=True)
            D_real_B = D_B(real_B, training=True)

            D_fake_A = D_A(fake_A, training=True)
            D_fake_B = D_B(fake_B, training=True)

            # Individual loss components
            g_loss_A2B = generator_loss(D_fake_B)
            g_loss_B2A = generator_loss(D_fake_A)

            cyc_loss_A = cycle_consistency_loss(real_A, recov_A)
            cyc_loss_B = cycle_consistency_loss(real_B, recov_B)

            id_loss_A = identity_loss(real_A, same_A)
            id_loss_B = identity_loss(real_B, same_B)

            # Could tweak these weights later if needed
            total_loss_G_A2B = g_loss_A2B + cyc_loss_A + id_loss_B
            total_loss_G_B2A = g_loss_B2A + cyc_loss_B + id_loss_A

            d_loss_A = discriminator_loss(D_real_A, D_fake_A)
            d_loss_B = discriminator_loss(D_real_B, D_fake_B)

        # Apply updates
        optimizer_GA.apply_gradients(zip(tape.gradient(total_loss_G_A2B, G_A2B.trainable_variables), G_A2B.trainable_variables))
        optimizer_GB.apply_gradients(zip(tape.gradient(total_loss_G_B2A, G_B2A.trainable_variables), G_B2A.trainable_variables))

        optimizer_DA.apply_gradients(zip(tape.gradient(d_loss_A, D_A.trainable_variables), D_A.trainable_variables))
        optimizer_DB.apply_gradients(zip(tape.gradient(d_loss_B, D_B.trainable_variables), D_B.trainable_variables))

        return {
            "gen_g_loss": g_loss_A2B,
            "gen_f_loss": g_loss_B2A,
            "cycle_loss_A": cyc_loss_A,
            "cycle_loss_B": cyc_loss_B,
            "disc_x_loss": d_loss_A,
            "disc_y_loss": d_loss_B,
            "identity_loss_A": id_loss_A,
            "identity_loss_B": id_loss_B
        }

    # Training loop begins
    for ep in range(EPOCHS):
        print(f"\nEpoch {ep + 1}/{EPOCHS} -------------------")
        start_time = time.time()
        epoch_totals = {k: 0.0 for k in all_losses.columns if k != "epoch"}
        batch_count = 0

        for real_A, real_B in tf.data.Dataset.zip((dataset_A, dataset_B)):
            losses = train_single_batch(real_A, real_B)
            for k in epoch_totals:
                epoch_totals[k] += float(losses[k])
            batch_count += 1

            if batch_count % 10 == 0:
                print(".", end="")  # Quick progress dot

        # Averaging out
        for k in epoch_totals:
            epoch_totals[k] /= batch_count

        epoch_totals["epoch"] = ep + 1
        all_losses = all_losses.append(epoch_totals, ignore_index=True)
        all_losses.to_csv(LOSS_LOG_PATH, index=False)

        # Save model + sample output every few epochs
        if (ep + 1) % 5 == 0:
            ckpt_path = manager.save()
            print(f"\nSaved checkpoint to: {ckpt_path}")

            # Preview generation
            try:
                preview_A = next(iter(dataset_A))
                preview_B = next(iter(dataset_B))
            except Exception as e:
                print(f"Preview generation failed: {e}")
                continue

            sample_fake_B = G_A2B(preview_A, training=False)
            sample_fake_A = G_B2A(preview_B, training=False)

            sample_dir = os.path.join(preview_dir_root, f"epoch_{ep + 1}")
            os.makedirs(sample_dir, exist_ok=True)
            save_image_pair(preview_A, sample_fake_B, sample_dir, idx=0)
            save_image_pair(preview_B, sample_fake_A, sample_dir, idx=1)

        print(f"\nEpoch {ep + 1} completed in {time.time() - start_time:.2f}s")

    # Save final models
    final_save_dir = os.path.join(CHECKPOINT_DIR, "final_generators")
    os.makedirs(final_save_dir, exist_ok=True)
    G_A2B.save_weights(os.path.join(final_save_dir, "generator_g.h5"))
    G_B2A.save_weights(os.path.join(final_save_dir, "generator_f.h5"))
    print("All done. Final model weights saved.")
