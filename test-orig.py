import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import pandas as pd

# Your dataset definition (reused)
from train import (
    SummerWinterDataset, 
    parse_args, 
    import_model_class_from_model_name_or_path, 
    load_datasets, 
    log_validation,
    compute_ssim,
    compute_lpips,
)




def main(args):
    args.split = 'test'
    args.resolution = 512
    args.num_validation_images = 1
    args.root_dir = "/iacl/pg23/shimeng/misc/DL2025/Project/data/summer+winter"
    args.split_result_file = "/iacl/pg23/shimeng/misc/DL2025/Project/dataset_split.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision=None)
    text_encoder = text_encoder_cls.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device)

    _, _, test_dataset =  load_datasets(args, tokenizer)
    
    metrics = []
    # checkpoints = [300, 450, 850, 1100, 1400] #base-depthmap
    # checkpoints = [2300, 4300, 4900, 5400, 5900] #base-depthmap-lpsis
    checkpoints = [1100, 2200, 3400, 4500, 5600] #base-depthmap-scrach

    loss_fn = lpips.LPIPS(net='alex')
    
    for i in checkpoints:
            
        checkpoint_path = "/iacl/pg23/shimeng/misc/DL2025/Project/condi-target_base-none_mixeprecision/checkpoint-" + str(i) +"/controlnet"
        
        
        controlnet = ControlNetModel.from_pretrained(checkpoint_path).to(device)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            tokenizer=tokenizer,
            safety_checker=None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_xformers_memory_efficient_attention()
        
        
        
        image_logs = []
        ssim_total, lpips_total, psnr_total, count = 0, 0, 0, 0
        for batch in tqdm(test_dataset, desc="Validating"):
            validation_condition = batch['source_as_condition']
            validation_prompt = batch['txt']
            target = batch['target']

            target_np = np.array(target).astype(np.float32) / 255.0
            images = []
            ssim_scores = []
            lpips_scores = []
            psnr_scores = []
            
            image = pipe(
                validation_prompt, validation_condition.unsqueeze(0), num_inference_steps=20, generator=None
            ).images[0]
                
            gen_np = np.array(image).astype(np.float32) / 255.0
            ssim_score = compute_ssim(gen_np, target_np)
            lpips_score = compute_lpips(gen_np, target_np, loss_fn)
            psnr_score = psnr(gen_np, target_np)
            images.append(image)
            ssim_scores.append(ssim_score)
            lpips_scores.append(lpips_score)
            psnr_scores.append(psnr_score)
            ssim_total += ssim_score
            lpips_total += lpips_score
            psnr_total += psnr_score
            count += 1
            image_log = batch
            image_log['gen_images'] = images
            image_log['ssim'] = np.mean(ssim_scores)
            image_log['lpips'] = np.mean(lpips_scores)
            image_log['psnr'] = np.mean(psnr_scores)
            image_logs.append(image_log)
        
        avg_ssim = ssim_total / count
        avg_lpips = lpips_total / count
        avg_psnr = psnr_total / count
        
        metrics.append({
            "checkpoint": i,
            "avg_ssim": avg_ssim,
            "avg_lpips": avg_lpips,
            "avg_psnr": avg_psnr
        })
        
        print(f"Checkpoint {i}:")
        print(f"Average SSIM: {avg_ssim}")
        print(f"Average LPIPS: {avg_lpips}")
        print(f"Average PSNR: {avg_psnr}")

        image_logs[0]['gen_images'][0].save(f"test_output_{checkpoint_path.split('/')[-3]}_{i}.png")
        
    pd.DataFrame(metrics).to_csv(f"{checkpoint_path.split('/')[-3]}_metrics.csv", index=False)

if __name__ == "__main__":
    args = argparse.Namespace()
    main(args)