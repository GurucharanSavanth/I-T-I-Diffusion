'''
Note: If the tester or developer using this code to implement there
own custom dataset this is the format to be followed through:

1) Build a Custom Vector DB with a pipeline to PyTorch or TensorFlow; 
2) Add a minimum of 32-128 layers; 
3) Tweak the temperature from 0-1 with decimal pointer changes of +0.01; 
4) Ensure dataset is cleaned, preprocessed, and labeled accurately with balanced classes; 
5) Store embeddings in a vector database with efficient indexing, compatible with PyTorch/TensorFlow; 
6) Experiment with different architectures (UNet, Transformer) and consider residual connections; 
7) Tune hyperparameters (learning rate, batch size, dropout) using grid or random search; 
8) Implement a scalable training pipeline with checkpoints and early stopping; 
9) Apply regularization techniques like weight decay or L2; 10) Define clear evaluation metrics (e.g., FID) and validate on a test set; 
11) Use advanced data augmentation (random cropping, rotation, CutMix, MixUp); 
12) Implement post-processing (denoising, sharpening, super-resolution) for output quality; 
13) Document all steps for reproducibility and use version control for code and datasets.

Currently, I am using my own dataset: "https://huggingface.co/Savanthgc/MyCustomDiffusionModel".
Author: Gurucharan S.
The data model I used is a 138B Live DB, trained on images from 2000-2019, with baseline data scraped from 
Google Images.
Please note that emails, PRs, or improvement requests are not accepted. 
If you need additional features, please build your own data model.

'''

# File: modify_image.py
import torch
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from PIL import Image
from diffusers import (
    StableDiffusionImageVariationPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler
)
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPSegProcessor, CLIPSegForImageSegmentation
)
from torchvision import transforms
from torchvision.utils import save_image
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import cv2
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from tqdm import tqdm
import joblib

# Initialize NLTK data
nltk.download(['punkt', 'stopwords'], quiet=True)

# GPU settings
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Caching
@lru_cache(maxsize=100)
def clean_prompt(prompt):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(prompt.lower())
    return ' '.join([w for w in word_tokens if w not in stop_words])


# Multi-model setup
@lru_cache(maxsize=1)
def load_models():
    models = {
        'variation': StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="main",
            torch_dtype=torch.float16,
            #use_safetensors=True
        ).to(device),
        'upscale': StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16,
        ).to(device),
        'inpaint': StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        ).to(device),
        'caption': BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device),
        'segment': CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(device)
    }

    processors = {
        'caption': BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large"),
        'segment': CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    }

    for model in models.values():
        if hasattr(model, 'enable_attention_slicing'):
            model.enable_attention_slicing()
        if hasattr(model, 'scheduler'):
            model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)

    return models, processors


# Dynamic hyperparameter tuning
def get_dynamic_hyperparameters(prompt, image):
    prompt_complexity = len(word_tokenize(prompt))
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    image_complexity = np.sum(edges > 0)

    num_inference_steps = max(50, min(150, prompt_complexity * 2 + image_complexity // 1000))
    guidance_scale = max(5.0, min(15.0, 7.5 + prompt_complexity * 0.1 + image_complexity * 0.0001))

    return num_inference_steps, guidance_scale


# Parallel image processing
def process_image_variation(variation_pipe, upscale_pipe, inpaint_pipe, preprocessed_image, prompt, num_inference_steps,
                            guidance_scale):
    with autocast():
        # Generate variation
        variation = variation_pipe(
            preprocessed_image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        # Upscale
        upscaled = upscale_pipe(
            prompt=prompt,
            image=variation,
            noise_level=20,
            num_inference_steps=num_inference_steps // 2
        ).images[0]

        # Inpaint (refine details)
        mask = Image.new('L', upscaled.size, 128)  # Gray mask for subtle inpainting
        inpainted = inpaint_pipe(
            prompt=prompt,
            image=upscaled,
            mask_image=mask,
            num_inference_steps=num_inference_steps // 2,
            guidance_scale=guidance_scale * 0.8
        ).images[0]

    return inpainted


# Main function
def modify_image(image_path, prompt, num_variations=1, batch_size=1):
    # Load models
    models, processors = load_models()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')

    # Generate caption
    inputs = processors['caption'](image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_caption = models['caption'].generate(**inputs)
    caption = processors['caption'].decode(generated_caption[0], skip_special_tokens=True)
    print(f"Generated caption: {caption}")

    # Preprocess prompt
    refined_prompt = clean_prompt(prompt)
    print(f"Refined Prompt: {refined_prompt}")

    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    preprocessed_image = preprocess(image).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)

    # Get dynamic hyperparameters
    num_inference_steps, guidance_scale = get_dynamic_hyperparameters(refined_prompt, image)

    # Generate variations in parallel
    all_images = []
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = []
        for _ in range(0, num_variations, batch_size):
            batch_prompts = [refined_prompt] * min(batch_size, num_variations - len(all_images))
            futures.extend([
                executor.submit(
                    process_image_variation,
                    models['variation'],
                    models['upscale'],
                    models['inpaint'],
                    preprocessed_image[i].unsqueeze(0),
                    prompt,
                    num_inference_steps,
                    guidance_scale
                )
                for i, prompt in enumerate(batch_prompts)
            ])

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating variations"):
            all_images.append(future.result())

    # Save results
    os.makedirs("outputs", exist_ok=True)
    for i, img in enumerate(all_images):
        img.save(f"outputs/result_{i + 1}.jpg")

    if len(all_images) > 1:
        save_image(torch.stack([transforms.ToTensor()(img) for img in all_images]), 'outputs/collage.jpg',
                   nrow=int(np.sqrt(len(all_images))))

    return len(all_images)
