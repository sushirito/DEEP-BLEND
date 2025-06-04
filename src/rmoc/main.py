import os

def get_image_files(directory):
    """
    Lists all .png/.jpg/.jpeg files in a given folder.
    Handy for verifying image availability.
    """
    return [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


if __name__ == "__main__":
    # Set base path
    project_root = os.getcwd()

    # Define input directories
    background_dir = os.path.join(project_root, "inputs", "backgrounds")
    foreground_dir = os.path.join(project_root, "inputs", "foregrounds")

    # Output path for all generated images
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # === FOPA-Guided Composition ===
    from fopa.compositer import generate_composite_images

    print("==> Running FOPA-guided compositing...")
    generate_composite_images(
        background_dir=background_dir,
        foreground_dir=foreground_dir,
        output_dir=output_dir
    )

    # === Baseline: Random Placement Composites ===
    from fopa.compositer import generate_base_case_composites

    rembg_output_dir = os.path.join(output_dir, "rembg")
    os.makedirs(rembg_output_dir, exist_ok=True)

    print("==> Generating baseline composites with random overlays...")
    generate_base_case_composites(
        background_dir=background_dir,
        foreground_dir=foreground_dir,
        output_dir=rembg_output_dir,
        num_backgrounds=100  # Feel free to change if needed
    )