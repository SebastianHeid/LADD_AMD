import argparse

import torch
import yaml
from box import Box
from core.data.eval_dataset import PromptImageDataset
from torch.utils.data import DataLoader
from torchmetrics.multimodal import CLIPScore
from tqdm import tqdm


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

    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()

    dataset = PromptImageDataset(
        config.generated_images_path,
        config.dataset.prompts.path,
        img_height=224,
        img_width=224,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    score = 0

    with torch.no_grad():
        for image, prompt in tqdm(dataloader):
            image = image.cuda()
            image = (image * 255).clamp(0, 255).byte()
            prompt = prompt[0]
            current_score = clip_score(image, [prompt]) / 100
            score += current_score

            del image, current_score
            torch.cuda.empty_cache()

        print("CLIP Score: ", score / dataset.__len__())
