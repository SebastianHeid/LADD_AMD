import argparse
import random

import numpy as np
import torch
import yaml
from box import Box
from core.data.eval_dataset import PromptFolderDataset
from diffusers import (
    AutoPipelineForText2Image,
    ConsistencyModelPipeline,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--config_path",
        type=str,
        const=True,
        default="/export/home/sheid/Adversarial-Diffusion-Distillation/config/config_eval_5.yaml",
        nargs="?",
        help="Path to config.yaml file",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        config = Box(yaml.safe_load(file))

    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if config.distill_method == "LADD":
        scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler"
        )
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", scheduler=scheduler
        )
        pipeline.unet.load_state_dict(torch.load(config.unet.checkpoints_path))
    if config.distill_method == "LADD_4":

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base"
        )
        pipeline.unet.load_state_dict(torch.load(config.unet.checkpoints_path))
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )

    if config.distill_method == "SDXL":
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
    elif config.distill_method == "ADD_4_step":
        if config.scheduler == "DDPM":
            scheduler = DDPMScheduler.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler"
            )
        elif config.scheduler == "ConsistencySampler":
            pip = ConsistencyModelPipeline.from_pretrained(
                "openai/diffusers-cd_imagenet64_l2", torch_dtype=torch.float16
            )
            scheduler = pip.scheduler
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", scheduler=scheduler
        )
        pipeline.unet.load_state_dict(
            torch.load(config.unet.checkpoints_path)["state_dict"]
        )

    pipeline.to("cuda")

    if config.dataset.prompts.name == "coco_val":
        dataset = PromptFolderDataset(config.dataset.prompts.path)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    pipeline.to("cuda")
    for img_name, prompt in tqdm(dataloader):
        prompt = prompt[0]
        if config.distill_method == "LADD" and config.num_inference_steps == 1:
            img = pipeline(
                prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
                timesteps=[999],
            ).images[0]
        elif config.distill_method == "LADD_4" and config.num_inference_steps == 4:
            print(pipeline.scheduler.timesteps)
            img = pipeline(
                prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
                timesteps=[999, 749, 499, 249],
            ).images[0]
        elif config.distill_method == "ADD_4_step":
            img = pipeline(
                prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
                timesteps=[999, 749, 499, 249],
            ).images[0]
        elif config.distill_method == "SDXL":
            img = pipeline(prompt).images[0]
        else:
            img = pipeline(prompt)
        img.save(config.generated_images_path + img_name[0] + ".png")
