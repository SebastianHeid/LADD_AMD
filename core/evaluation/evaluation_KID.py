import os

import numpy as np
import torch
import yaml
from box import Box
from PIL import Image
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
from torchvision import transforms
from tqdm import tqdm


def load_and_resize_image(image_path, target_size=None):
    """Load an image and optionally resize it to target_size (width, height)."""
    img = Image.open(image_path).convert("RGB")

    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img).unsqueeze(0)


def extract_features(dataloader, model, device, max_images=None):
    """Extract features from images using Inception v3"""
    features = []
    total_processed = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)

            # Forward pass through Inception v3
            try:
                pred = model(batch)

                # Handle Inception v3 output (returns tuple)
                if isinstance(pred, tuple):
                    pred = pred[0]  # Take the main output

                # Ensure we have 2D features (batch_size × features)
                if pred.dim() == 4:  # If we got spatial features (batch × C × H × W)
                    pred = torch.flatten(pred, start_dim=1)
                elif pred.dim() == 3:  # (batch × C × L)
                    pred = torch.flatten(pred, start_dim=1)

                features.append(pred.cpu().numpy())
                total_processed += batch.size(0)

                if max_images and total_processed >= max_images:
                    break

            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

    if not features:
        raise ValueError("No features were extracted - check your input data")

    features = np.concatenate(features, axis=0)

    # Ensure we have proper 2D features
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features but got {features.ndim}D array")

    return features


def kernel_inception_distance(features1, features2, degree=3, gamma=None, coef0=1):
    """Calculate Kernel Inception Distance between two sets of features"""
    # Input validation
    if features1.shape[0] == 0 or features2.shape[0] == 0:
        raise ValueError("Empty feature arrays provided")

    if features1.shape[1] != features2.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {features1.shape[1]} vs {features2.shape[1]}"
        )

    # Default gamma value
    if gamma is None:
        gamma = 1.0 / features1.shape[1]

    try:
        # Compute kernel matrices
        K_XX = polynomial_kernel(features1, degree=degree, gamma=gamma, coef0=coef0)
        K_YY = polynomial_kernel(features2, degree=degree, gamma=gamma, coef0=coef0)
        K_XY = polynomial_kernel(
            features1, features2, degree=degree, gamma=gamma, coef0=coef0
        )

        # Calculate KID
        kid = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
        return float(kid)
    except Exception as e:
        raise RuntimeError(f"KID calculation failed: {str(e)}")


def calculate_kid(
    directory1,
    directory2,
    resize_strategy="first",  # 'first', 'second', 'max', or 'min'
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=50,
    max_images=None,
    subsample_size=1000,
    degree=3,
    gamma=None,
    coef0=1,
):
    """
    Calculate KID scores between corresponding images in two directories,
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
        batch_size (int): Batch size for feature extraction
        max_images (int): Maximum number of images to use (None for all)
        subsample_size (int): Number of samples to use for KID calculation
        degree (int): Degree for polynomial kernel
        gamma (float): Gamma for polynomial kernel (None for 1/n_features)
        coef0 (float): Coef0 for polynomial kernel

    Returns:
        dict: Dictionary containing KID score and statistics
    """
    # Load Inception v3 model
    inception = torch.hub.load(
        "pytorch/vision:v0.10.0", "inception_v3", pretrained=True
    )
    inception.fc = torch.nn.Identity()  # Remove final classification layer
    inception = inception.to(device)
    inception.eval()

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

    # Determine target size by checking first pair
    if resize_strategy in ["first", "second", "max", "min"]:
        img1_path = os.path.join(directory1, files1[0])
        img2_path = os.path.join(directory2, files2[0])

        img1_pil = Image.open(img1_path).convert("RGB")
        img2_pil = Image.open(img2_path).convert("RGB")

        size1 = img1_pil.size
        size2 = img2_pil.size

        if resize_strategy == "first":
            target_size = size1
        elif resize_strategy == "second":
            target_size = size2
        elif resize_strategy == "max":
            target_size = (max(size1[0], size2[0]), max(size1[1], size2[1]))
        elif resize_strategy == "min":
            target_size = (min(size1[0], size2[0]), min(size1[1], size2[1]))

        print(f"Resizing all images to: {target_size}")
    else:
        target_size = None

    # Create dataloaders
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, directory, files, target_size=None):
            self.directory = directory
            self.files = files
            self.target_size = target_size

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img_path = os.path.join(self.directory, self.files[idx])
            img = load_and_resize_image(img_path, self.target_size)
            return img.squeeze(0)

    dataset1 = ImageDataset(directory1, files1, target_size)
    dataset2 = ImageDataset(directory2, files2, target_size)

    dataloader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=False, num_workers=4
    )
    dataloader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Extract features
    features1 = extract_features(dataloader1, inception, device, max_images)
    features2 = extract_features(dataloader2, inception, device, max_images)

    # Subsample features
    if subsample_size:
        rng = np.random.RandomState(42)
        if len(features1) > subsample_size:
            features1 = rng.choice(features1, subsample_size, replace=False)
        if len(features2) > subsample_size:
            features2 = rng.choice(features2, subsample_size, replace=False)

    # Calculate KID
    kid_value = kernel_inception_distance(
        features1, features2, degree=degree, gamma=gamma, coef0=coef0
    )

    return {
        "kid": kid_value,
        "num_images1": len(features1),
        "num_images2": len(features2),
        "feature_dim": features1.shape[1],
        "subsample_size": subsample_size,
        "resize_strategy": resize_strategy,
        "target_size": target_size,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate KID between two image directories"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        const=True,
        default="/export/home/sheid/LADD_AMD/config/config_eval_1.yaml",
        nargs="?",
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="second",
        choices=["first", "second", "max", "min"],
        help="Resize strategy for different image sizes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--subsample_size",
        type=int,
        default=5000,
        help="Number of samples to use for KID calculation",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        config = Box(yaml.safe_load(file))

    # Validate directories
    if not os.path.isdir(config.real_images_path):
        raise ValueError(f"Directory not found: {config.real_images_path}")
    if not os.path.isdir(config.generated_images_path):
        raise ValueError(f"Directory not found: {config.generated_images_path}")

    # Calculate KID
    print(
        f"\nCalculating KID between {config.real_images_path} and {config.generated_images_path}"
    )
    print(f"Resize strategy: {args.resize}")
    results = calculate_kid(
        config.real_images_path,
        config.generated_images_path,
        resize_strategy=args.resize,
        batch_size=args.batch_size,
        subsample_size=args.subsample_size,
    )

    # Print summary
    print("\nKID Results:")
    print(
        f"Directories compared: {config.real_images_path} vs {config.generated_images_path}"
    )
    print(
        f"Number of images (real/generated): {results['num_images1']}/{results['num_images2']}"
    )
    print(f"Feature dimension: {results['feature_dim']}")
    print(f"Subsample size: {results['subsample_size']}")
    print(f"Resize strategy: {results['resize_strategy']}")
    print(f"Target size: {results['target_size']}")
    print(f"\nKernel Inception Distance (KID): {results['kid']:.6f}")
