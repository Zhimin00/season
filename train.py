import torch
from PIL import Image
from torchvision import transforms
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import argparse
import os
import lpips
from skimage.metrics import structural_similarity as ssim
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import DDPMScheduler
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.controlnet.train()

class SeasonalDataset(Dataset):
    def __init__(self, src_dir, tgt_dir, canny_dir, transform=None):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.canny_dir = canny_dir
        self.image_names = sorted(os.listdir(src_dir))
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        src = Image.open(os.path.join(self.src_dir, img_name)).convert("RGB")
        tgt = Image.open(os.path.join(self.tgt_dir, img_name)).convert("RGB")
        canny = Image.open(os.path.join(self.canny_dir, img_name)).convert("L")

        return {
            "pixel_values": self.transform(tgt),           # 目标图像
            "condition_image": self.transform(canny),       # Canny 边图
            "prompt": "a {season} scene with buildings"     # 简单 prompt，你也可以用 caption
        }

from torch.optim import AdamW

dataset = SeasonalDataset("seasonal_dataset/summer", "seasonal_dataset/winter", "seasonal_dataset/canny")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = AdamW(pipe.controlnet.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch["pixel_values"].to("cuda")
        control = batch["condition_image"].to("cuda")
        prompt = batch["prompt"]

        with torch.autocast("cuda"):
            loss = pipe(
                prompt=prompt,
                image=images,
                control_image=control,
                return_loss=True
            ).loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Inference
result = pipe("a winter scene", control_image=canny_img).images[0]
result.save("output.jpg")
