import os

def get_image_files(directory: str):
    """
    Return a list of all .png/.jpg/.jpeg file paths under `directory`.
    """
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


if __name__ == "__main__":
    project_root = os.getcwd()

    # Input folders
    background_dir = os.path.join(project_root, "inputs", "backgrounds")
    foreground_dir = os.path.join(project_root, "inputs", "foregrounds")

    # Output folders
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    #Generate FOPA-guided composites
    from fopa.compositer import generate_composite_images

    print("==> Generating FOPA composites ...")
    generate_composite_images(
        background_dir=background_dir,
        foreground_dir=foreground_dir,
        output_dir=output_dir
    )

    # Generate random base-case composites (use the same `foreground_dir`)
    rembg_output_dir = os.path.join(output_dir, "rembg")
    os.makedirs(rembg_output_dir, exist_ok=True)

    from fopa.compositer import generate_base_case_composites

    print("==> Generating base-case random composites ...")
    generate_base_case_composites(
        background_dir=background_dir,
        foreground_dir=foreground_dir,
        output_dir=rembg_output_dir,
        num_backgrounds=100
    )
