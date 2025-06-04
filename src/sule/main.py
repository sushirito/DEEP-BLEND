import argparse
import os

from .train import train_cycle_gan
from .inference import load_weights_into_generator, save_test_predictions


def parse_args():
    """
    Command-line interface for training or testing the CycleGAN model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "test"], help="Choose between training or testing.")
    parser.add_argument("--data_A", type=str, help="Path to domain A images")
    parser.add_argument("--data_B", type=str, help="Path to domain B images")
    parser.add_argument("--output", type=str, help="Directory to save outputs (checkpoints or predictions)")
    parser.add_argument("--weights", type=str, help="Path to generator weights (.h5), used in test mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        # Training requires both datasets and an output folder
        if not (args.data_A and args.data_B and args.output):
            raise ValueError("Training mode requires --data_A, --data_B, and --output.")

        os.makedirs(args.output, exist_ok=True)
        train_cycle_gan(args.data_A, args.data_B, args.output)

    elif args.mode == "test":
        # Testing requires weights and an input folder
        if not (args.weights and args.data_A and args.output):
            raise ValueError("Testing mode requires --weights, --data_A, and --output.")

        # Load the pre-trained generator and run inference
        generator = load_weights_into_generator(args.weights)
        save_test_predictions(generator, args.data_A, args.output)
