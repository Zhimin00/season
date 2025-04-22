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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_fn = lpips.LPIPS(net='alex')

def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0, multichannel=True)

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
    seg.save('seg_mask.jpg')
    return seg

def get_canny(input_path):
    img = cv2.imread(input_path)
    img = cv2.resize(img, (512, 512))  # Resize to fit SD
    edges = cv2.Canny(img, 100, 200)
    canny_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    canny_img = Image.fromarray(canny_rgb)
    canny_img.save('canny.jpg')
    return canny_img

def inference(input_path, output_path, prompt):
    canny_control = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
    seg_control = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[canny_control, seg_control],
        torch_dtype=torch.float16,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    canny_img = get_canny(input_path)

    processor, model = load_segformer()
    image = load_image(input_path)
    seg_img = get_seg_mask(image, processor, model)
    
    result = pipe(
        prompt=prompt,
        image=[canny_img, seg_img],
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    result.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="test_winter.jpg")
    parser.add_argument("--output", type=str, default="output_1.jpg")
    parser.add_argument("--prompt", type=str, default="a photo of the campus in spring")
    args = parser.parse_args()

    inference(args.input, args.output, args.prompt)