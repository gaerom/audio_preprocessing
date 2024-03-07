""" MLP를 거쳐 mapping 된 text embedding을 input으로 사용하여 diffusion model output 확인해보기 
https://huggingface.co/stabilityai/stable-diffusion-2-1-base """

import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import os

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler # diffusion



class EmbeddingToImageDiffusion:
    def __init__(self, diffusion_model_id, device):
        self.device = device
        self.diffusion_model = StableDiffusionPipeline.from_pretrained(diffusion_model_id, torch_dtype=torch.float16).to(device)

    def generate_images(self, embeddings):
        embeddings_tensor = torch.tensor(embeddings).float().to(self.device)
        generated_images = self.diffusion_model(prompt_embeds=embeddings_tensor).images
        return generated_images

    def save_images(self, images, output_dir, prefix="generated_image"):
        os.makedirs(output_dir, exist_ok=True)
        for i, img in enumerate(images):
            img_path = os.path.join(output_dir, f"{prefix}_{i}.png")
            img.save(img_path)
            print(f"Image saved to {img_path}")
