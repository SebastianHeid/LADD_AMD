import os

import lpips
import numpy as np
import torch
import yaml
from box import Box
from PIL import Image
from torchvision import transforms


def load_and_resize_image(image_path, target_size=None):
    """Load an image and optionally resize it to target_size (width, height)."""
    img = Image.open(image_path).convert("RGB")

    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(img).unsqueeze(0)


def calculate_lpips(
    directory1,
    directory2,
    resize_strategy="first",  # 'first', 'second', 'max', or 'min'
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Calculate LPIPS scores between corresponding images in two directories,
    handling different image sizes by resizing.

    Args:
        directory1 (str): Path to first directory of images
        directory2 (str): Path to second directory of images
        resize_strategy (str): How to handle different sizes:
            'first' - resize second image to first image's dimensions
            'second' - resize first image to second image's dimensions
            'max' - resize both to maximum dimensions
            'min' - resize both to minimum dimensions
        device (str): Device to use for computation ('cuda' or 'cpu')

    Returns:
        dict: Dictionary containing mean LPIPS score and individual scores
    """
    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net="alex").to(device)

    # Get sorted list of files in each directory
    files1 = sorted(
        [
            f
            for f in os.listdir(directory1)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    )
    files2 = sorted(
        [
            f
            for f in os.listdir(directory2)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    )

    if len(files1) != len(files2):
        print(
            f"Warning: Directory 1 has {len(files1)} images, Directory 2 has {len(files2)} images"
        )

    # Calculate LPIPS for each pair
    lpips_scores = []
    size_mismatches = 0
    resized_pairs = 0

    for f1, f2 in zip(files1, files2):
        img1_path = os.path.join(directory1, f1)
        img2_path = os.path.join(directory2, f2)

        try:
            # Load images to check sizes
            img1_pil = Image.open(img1_path).convert("RGB")
            img2_pil = Image.open(img2_path).convert("RGB")

            size1 = img1_pil.size
            size2 = img2_pil.size

            # Determine target size based on strategy
            if size1 != size2:
                size_mismatches += 1

                if resize_strategy == "first":
                    target_size = size1
                elif resize_strategy == "second":
                    target_size = size2
                elif resize_strategy == "max":
                    target_size = (max(size1[0], size2[0]), max(size1[1], size2[1]))
                elif resize_strategy == "min":
                    target_size = (min(size1[0], size2[0]), min(size1[1], size2[1]))
                else:
                    raise ValueError(f"Unknown resize strategy: {resize_strategy}")

                resized_pairs += 1
                # print(f"Resizing {f1} ({size1}) and {f2} ({size2}) to {target_size}")
            else:
                target_size = None

            # Load images with optional resizing
            img1 = load_and_resize_image(img1_path, target_size).to(device)
            img2 = load_and_resize_image(img2_path, target_size).to(device)

            # Calculate LPIPS
            with torch.no_grad():
                score = loss_fn(img1, img2)

            lpips_scores.append(score.item())
        # print(f"{f1} vs {f2}: {score.item():.4f}")

        except Exception as e:
            print(f"Error processing {f1} and {f2}: {str(e)}")
            continue

    if not lpips_scores:
        raise ValueError("No valid image pairs were processed")

    mean_score = np.mean(lpips_scores)
    std_score = np.std(lpips_scores)

    print("\nStatistics:")
    print(f"Total image pairs processed: {len(lpips_scores)}")
    print(f"Pairs with size mismatches: {size_mismatches}")
    print(f"Pairs that were resized: {resized_pairs}")
    print(f"Mean LPIPS score: {mean_score:.4f} ± {std_score:.4f}")

    return {
        "mean": mean_score,
        "std": std_score,
        "scores": lpips_scores,
        "num_pairs": len(lpips_scores),
        "size_mismatches": size_mismatches,
        "resized_pairs": resized_pairs,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate LPIPS between two image directories"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        const=True,
        default="/export/home/sheid/Adversarial-Diffusion-Distillation/config/config_eval_ref_sd.yaml",
        nargs="?",
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--resize",
        type=str,
    ",
        choices=["first", "second", "max", "min"],
        help="Resize strategy for different image sizes",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        config = Box(yaml.safe_load(file))

    # Validate directories
    if not os.path.isdir(config.real_images_path):
        raise ValueError(f"Directory not found: {config.real_images_path}")
    if not os.path.isdir(config.generated_images_path):
        raise ValueError(f"Directory not found: {config.generated_images_path}")

    # Calculate LPIPS
    print(
        f"\nCalculating LPIPS between {config.real_images_path} and {config.generated_images_path}"
    )
    print(f"Resize strategy: {args.resize}")
    results = calculate_lpips(
        config.real_images_path,
        config.generated_images_path,
        resize_strategy=args.resize,
    )

    # Print summary
    print("\nFinal Summary:")
    print(
        f"Directories compared: {config.real_images_path} vs {config.generated_images_path}"
    )
    print(f"Number of image pairs: {results['num_pairs']}")
    print(f"Pairs with different sizes: {results['size_mismatches']}")
    print(f"Pairs that were resized: {results['resized_pairs']}")
    print(f"Mean LPIPS score: {results['mean']:.4f}")
    print(f"Standard deviation: {results['std']:.4f}")
