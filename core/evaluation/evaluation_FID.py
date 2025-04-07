import argparse
import random

import numpy as np
import torch
import yaml
from box import Box
from cleanfid import fid


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--config_path",
        type=str,
        const=True,
        default="/export/home/sheid/Adversarial-Diffusion-Distillation/config/config_eval_ref_sd.yaml",
        nargs="?",
        help="Path to config.yaml file",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        config = Box(yaml.safe_load(file))

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    fid_score = fid.compute_fid(config.real_images_path, config.generated_images_path)

    print(f"FID score: {fid_score.item()}")
