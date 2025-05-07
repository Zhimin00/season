import torch
from PIL import Image
from torchvision import transforms
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import argparse
import os
import lpips
from skimage.metrics import structural_similarity as ssim
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import numpy as np
import json


device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_fn = lpips.LPIPS(net='alex')

def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0, channel_axis=-1)


def compute_lpips(img1, img2, loss_fn):
    img1_t = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    img2_t = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return loss_fn(img1_t, img2_t).item()


def load_segformer():
    processor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(device)
    return processor, model


def get_seg_mask(img, processor, model):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    seg = outputs.logits.argmax(dim=1)[0].cpu().numpy()
    seg = Image.fromarray((seg * 10).astype("uint8")).resize((512, 512)).convert("RGB")
    #seg.save('seg_mask.jpg')
    return seg


def get_canny(img):
    edges = cv2.Canny(np.asarray(img), 100, 200)
    canny_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    canny_img = Image.fromarray(canny_rgb)
    #canny_img.save('canny.jpg')
    return canny_img


def inference(batch, output_path, prompt):
    
    canny_img = batch['canny'][0]#get_canny(img)

    
    seg_img = batch['seg'][0]#get_seg_mask(image, processor, model)

    result = pipe(
        prompt=prompt,
        image=[canny_img, seg_img],
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    result.save(output_path)
    return result


class SummerWinterDataset(Dataset):
    def __init__(self, root_dir, split_result_file, split='train', transform=None):
        self.summer_dir = os.path.join(root_dir, 'summer')
        self.winter_dir = os.path.join(root_dir, 'winter')
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        self.img_pairs = []
        split_result = []
        with open(split_result_file, 'r') as file:
            for line in file:
                split_result.append(json.loads(line))
        for img_name in sorted(os.listdir(self.summer_dir)):
            if {"file_name": img_name, 'location': root_dir.split('/')[-1], 'split': split} not in split_result:
                continue
            summer_img_path = os.path.join(self.summer_dir, img_name)
            winter_img_path = os.path.join(self.winter_dir, img_name.replace('s', 'w'))
            assert os.path.exists(summer_img_path) and os.path.exists(winter_img_path)
            if os.path.exists(summer_img_path) and os.path.exists(winter_img_path):
                self.img_pairs.append((summer_img_path, winter_img_path, img_name))
        processor, model = load_segformer()
        self.model = model
        self.processor = processor
        # print(len(self.img_pairs))

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        summer_img_path, winter_img_path, img_name = self.img_pairs[idx]
        summer_img = Image.open(summer_img_path).convert("RGB")
        winter_img = Image.open(winter_img_path).convert("RGB")
        canny_img = get_canny(winter_img)
        seg_img = get_seg_mask(winter_img, self.processor, self.model)

        if self.transform:
            summer_img = self.transform(summer_img)
            winter_img = self.transform(winter_img)
            canny_img = self.transform(canny_img)
            seg_img = self.transform(seg_img)

        return {
        'winter': winter_img,
        'summer': summer_img,
        'canny': canny_img,
        'seg': seg_img,
        'img_name': img_name,
        }

def inference_location(root_dir, split_result_file, output_base, prompt, csv_path):
    dataset = SummerWinterDataset(root_dir, split_result_file, split='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(output_base, exist_ok=True)
    results = []

    for i, batch in enumerate(dataloader):
        # summer_img_pil = Image.fromarray((summer_img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        # winter_img_np = winter_img[0].permute(1, 2, 0).numpy()
       # winter_img_pil = Image.fromarray((winter_img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        summer_img_np = batch['summer'][0].permute(1, 2, 0).numpy()
        img_name = img_name[0]
        output_path = os.path.join(output_base, img_name)

        gen_img_pil = inference(batch, output_path, prompt)
        gen_img_np = np.array(gen_img_pil).astype(np.float32) / 255.0
        # evaluate
        ssim_score = compute_ssim(gen_img_np, summer_img_np)
        lpips_score = compute_lpips(gen_img_np, summer_img_np, loss_fn)
        print(f"[{i}] SSIM: {ssim_score:.4f}, LPIPS: {lpips_score:.4f}")
        results.append((img_name, ssim_score, lpips_score))

    # 保存evaluation结果
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image_Name", "SSIM", "LPIPS"])
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--split_result_file", type=str, default="./dataset_split.jsonl")
    parser.add_argument("--output_base", type=str, default="./output")
    parser.add_argument("--prompt", type=str, default="a photo of the campus in summer")
    parser.add_argument("--csv_path", type=str, default="./evaluation")
    args = parser.parse_args()
    canny_control = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
    seg_control = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[canny_control, seg_control],
        torch_dtype=torch.float16,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    processor, model = load_segformer()
    for dir in os.listdir(args.root_dir):
        dir_path = os.path.join(args.root_dir, dir)
        if not os.path.isdir(dir_path):
            continue
        output_dir = os.path.join(args.output_base, dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        inference_location(dir_path, args.split_result_file, output_dir, args.prompt, os.path.join(args.csv_path, f"{dir.split('_')[-1]}.csv"))
