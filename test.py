# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn.functional as F
# from core.data.dataset import ADDDataset
# from diffusers import AutoencoderKL
# from PIL import Image

# train_dataset = ADDDataset(
#     "/export/data/vislearn/rother_subgroup/sheid/LAION_LADD_SDXL/", "summary.pkl"
# )

# for i in range(10):
#     latents, noises, text_embs = train_dataset.__getitem__(i)
#     latents = latents.unsqueeze(0)
#     print(latents.shape)
#     latents_downsampled = F.interpolate(
#         latents, size=(64, 64), mode="bilinear", align_corners=False
#     )

#     # Upsample zurück auf 128x128 für den SDXL-VAE
#     latents_upsampled = F.interpolate(
#         latents_downsampled, size=(128, 128), mode="bilinear", align_corners=False
#     )

#     latents = latents_upsampled.to("cuda")
#     vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to("cuda").eval()
#     with torch.no_grad():
#         image_reconstructed = vae.decode(latents / 0.18215).sample
#     image_reconstructed = (
#         image_reconstructed.clamp(-1, 1) + 1
#     ) / 2  # Normierung auf [0,1]
#     image_reconstructed = image_reconstructed.permute(0, 2, 3, 1).cpu().numpy()[0]

#     # Konvertiere das numpy-Array in ein PIL-Bild
#     image_pil = Image.fromarray((image_reconstructed * 255).astype(np.uint8))
#     image_pil.save("reconstructed_image_du_" + str(i) + ".png")

# import random

# import numpy as np
# import torch
# from diffusers import StableDiffusionXLPipeline

# # Set the seed for reproducibility
# seed_value = 42

# # For Python's random library
# random.seed(seed_value)

# # For NumPy
# np.random.seed(seed_value)

# # For PyTorch
# torch.manual_seed(seed_value)
# torch.cuda.manual_seed_all(seed_value)  # For all GPUs if using CUDA
# # Load the pre-trained SDXL pipeline
# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0"
# ).to("cuda")
# vae_sdxl = pipe.vae

# # Set the desired image resolution (512x512)
# height, width = 512, 512
# # height, width = 1024, 1024
# # Generate latents manually (this is the internal latent space of SDXL)
# latents = torch.randn(
#     (1, 4, height // 8, width // 8), device="cuda"
# )  # 4 channels, resolution downscaled by 8
# print("Latent shape:", latents.shape)  # Should print torch.Size([1, 4, 64, 64])

# # Use the pipeline to generate an image
# prompt = "A real image of a cat."
# image = pipe(prompt, num_inference_steps=50, height=height, width=width).images[0]


# # Optionally save the image
# image.save("generated_image_512x512.png")
# import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

# from PIL import Image
# from torch.nn.functional import interpolate

# # Set the seed for reproducibility
# seed_value = 42
# random.seed(seed_value)
# np.random.seed(seed_value)
# torch.manual_seed(seed_value)
# torch.cuda.manual_seed_all(seed_value)

# # Load SDXL pipeline
# pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
#     "/export/scratch/sheid/models/sd_xl/",
# ).to("cuda")
# vae_sdxl = pipe_sdxl.vae  # SDXL's VAE

# # Load SD2.1 pipeline
# pipe_sd21 = StableDiffusionPipeline.from_pretrained(
#     "/export/scratch/sheid/models/stable-diffusion-2-1-base"
# ).to("cuda")
# vae_sd21 = pipe_sd21.vae  # SD2.1's VAE

# # Set image resolution
# height, width = 512, 512

# # Generate an image using SDXL
# prompt = "A highly realistic photo of a sport car in the  mountains."

# with torch.no_grad():
#     # Generate an image using SDXL
#     # prompt = "A highly realistic photo of a mountain with a sea."
#     image_sdxl = pipe_sdxl(prompt, num_inference_steps=50).images[0]
#     image_sdxl.save("generated_sdxl.png")
#     # Downsample image using bilinear interpolation (using PIL)
#     # downsampled_image = image_sdxl.resize((512, 512), Image.BILINEAR)

#     # Alternatively, if you prefer to use PyTorch's interpolate function
#     # downsampled_image = F.interpolate(image_tensor, size=(512, 512), mode='bilinear', align_corners=False)

#     # Save the downsampled image
#     # downsampled_image.save("downsampled_sdxl.png")
#     # Convert image to latents using SDXL's VAE
#     image_tensor = (
#         torch.tensor(np.array(image_sdxl)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#     )
#     image_tensor = (image_tensor - 0.5) * 2  # Normalize to [-1, 1]
#     image_tensor = image_tensor.to("cuda")

#     latents_sdxl = vae_sdxl.encode(image_tensor).latent_dist.sample() * 0.18215
#     latents_sdxl = F.interpolate(
#         latents_sdxl, size=(64, 64), mode="bilinear", align_corners=False
#     )
#     print("SDXL Latent shape:", latents_sdxl.shape)

#     # Perform DDIM inversion (Adding noise to latents to simulate a specific diffusion step)
#     scheduler = pipe_sd21.scheduler
#     scheduler.set_timesteps(50)  # Define denoising steps
#     noise_level = torch.tensor([999])  # Midway noise level

#     noisy_latents = scheduler.add_noise(
#         latents_sdxl, torch.randn_like(latents_sdxl), noise_level
#     )
#     print(noisy_latents.shape)  # Verify actual shape
#     print(pipe_sd21.unet.config.sample_size)  # Expected input size (e.g., 64 for SD2.1)
#     # Generate final image using SD2.1 with DDIM denoising
#     image_final = pipe_sd21(
#         prompt, latents=noisy_latents, num_inference_steps=50
#     ).images[0]
#     image_final.save("final_generated_sd21.png")

# Lade das Stable Diffusion 2.1 Modell
pipe_sd21 = StableDiffusionPipeline.from_pretrained(
    "/export/scratch/sheid/models/stable-diffusion-2-1-base"
).to("cuda")

unet = UNet2DConditionModel.from_pretrained(
    "/export/scratch/sheid/models/stable-diffusion-2-1-base", subfolder="unet"
).to("cuda")
# tokenizer = pipe_sd21.tokenizer
# text_encoder = pipe_sd21.text_encoder

# # Beispiel-Prompt
# prompt = "A scenic landscape with mountains and a river."

# # Tokenisiere den Prompt und berechne die Text-Embeddings
# text_inputs = tokenizer(
#     prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77
# )
# input_ids = text_inputs.input_ids.to("cuda")
# text_embeddings = text_encoder(input_ids)[0]  # Shape: (1, 77, 1024) für SD2.1

# # Erstelle zufällige latents
# latents = torch.randn((1, 4, 128, 128), device="cuda")

# # Führe den UNet-Vorhersageschritt aus
# timestep = torch.tensor([50], device="cuda")  # Beispiel-Timestep
# pred1 = unet(latents, timestep, encoder_hidden_states=text_embeddings).sample
# Sollte (1, 4, 64, 64) sein
total_params = sum(p.numel() for p in unet.parameters())

# Print result
print(f"Total parameters in U-Net: {total_params:,}")
pipe_sd21.unet = unet
# img = pipe_sd21("A very nice garden.").images[0]
# img.save("architecture.png")

last_up_block = unet.up_blocks[-1]
print(last_up_block)
# if hasattr(last_up_block, "resnets") and len(last_up_block.resnets) > 0:
#     # Entferne das letzte ResNet, behalte aber alle anderen
#     new_resnets = nn.ModuleList(last_up_block.resnets[:-1])
#     last_up_block.resnets = new_resnets
#     print("------------------------------------------------------------")
#     print(last_up_block)

#     # Erzwinge eine Neuinitialisierung der Parameter
#     for param in last_up_block.parameters():
#         if param.requires_grad:
#             param.data = param.data.clone()

# # img = pipe_sd21("A very nice garden.").images[0]
# # img.save("removed_architecture.png")
# # print(pred.shape)

# total_params = sum(p.numel() for p in unet.parameters())

# # Print result
# print(f"Total parameters in U-Net: {total_params:,}")

# # Load the original U-Net
# unet = UNet2DConditionModel.from_pretrained(
#     "/export/scratch/sheid/models/stable-diffusion-2-1-base", subfolder="unet"
# )
# print(f"Number of upsampling blocks: {len(unet.up_blocks)}")

# first_up_block = unet.up_blocks[1]
# print(first_up_block)
# for name, module in first_up_block.named_children():
#     print(f"Layer: {name}, Type: {type(module)}")

# unet = UNet2DConditionModel.from_pretrained(
#     "/export/scratch/sheid/models/stable-diffusion-2-1-base", subfolder="unet"
# )

# # Get the last upsampling block
# last_up_block = unet.up_blocks[-1]
# print(last_up_block.resnets)
# # Remove the last ResNet block in the last upsampling block
# if hasattr(last_up_block, "resnets") and len(last_up_block.resnets) > 0:
#     last_up_block.resnets = nn.ModuleList(
#         last_up_block.resnets[:-1]
#     )  # Remove the last ResNet block

# print("Last ResNet block removed from final upsampling block!")
# print(last_up_block.resnets)
# print("--------------------------------------------------------------")
# print(unet.conv_out)
# Count ResNet blocks
# resnet_blocks = []
# for block in unet.down_blocks + unet.up_blocks + [unet.mid_block]:
#     for layer in block.children():
#         if isinstance(
#             layer, torch.nn.ModuleList
#         ):  # ResNet blocks are inside ModuleLists
#             for sublayer in layer:
#                 if "resnets" in sublayer.__class__.__name__.lower():
#                     resnet_blocks.append(sublayer)

# print(f"Total ResNet Blocks: {len(resnet_blocks)}")
