import json
import os

import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, img_height=None, img_width=None):
        self.folder_path = folder_path
        self.img_height = img_height
        self.img_width = img_width
        self.image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = ToTensor()(Image.open(image_path).convert("RGB"))
        if (self.img_height is not None) and (self.img_width is not None):
            image = self.transform(image)
        return image

    def transform(self, image):
        return F.interpolate(image.unsqueeze(0), size=(self.img_height, self.img_width), mode="bilinear", align_corners=False).squeeze(0)


class PromptFolderDataset(Dataset):
    def __init__(self, prompt_path):
        self.prompt_path = prompt_path
        with open(self.prompt_path, "r") as file:
            self.prompts = json.load(file)
        self.prompts_list = sorted(list(self.prompts.items()))

    def __len__(self):
        return len(self.prompts_list)

    def __getitem__(self, idx):
        _, prompt = self.prompts_list[idx]
        return _, prompt


class PromptImageDataset(Dataset):
    def __init__(self, generated_image_path, prompt_path, img_height=None, img_width=None):
        self.prompt_path = prompt_path
        self.generated_image_path = generated_image_path
        self.img_height = img_height
        self.img_width = img_width

        with open(self.prompt_path, "r") as file:
            self.prompts = json.load(file)
        self.prompts_list = sorted(list(self.prompts.items()))

    def __len__(self):
        return len(self.prompts_list)

    def __getitem__(self, idx):
        img_name, prompt = self.prompts_list[idx]
        image = ToTensor()(Image.open(self.generated_image_path + img_name + ".png").convert("RGB"))
        if (self.img_height is not None) and (self.img_width is not None):
            image = self.transform(image)
        return image, prompt

    def transform(self, image):
        return F.interpolate(image.unsqueeze(0), size=(self.img_height, self.img_width), mode="bilinear", align_corners=False).squeeze(0)


# path = "/export/home/sheid/Adversarial-Diffusion-Distillation/images/coco/"
# path = "/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/val2017/"
# path_prompts = "/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val.json"
# dataset = PromptImageDataset(path_prompts, path, img_height=512, img_width=512)

# img, prompt = dataset.__getitem__(3)
# print(prompt)
# img = ToPILImage()(img)
# img.save("/export/home/sheid/Adversarial-Diffusion-Distillation/images/coco/img.png")
# d = PromptFolderDataset(path_prompts)
# print(d.__getitem__(0))
# print(d.__len__())
