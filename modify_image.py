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

import torch
from PIL import Image
from diffusers import StableDiffusionImageVariationPipeline
from torchvision import transforms

def modify_image(image_path, prompt, strength, num_variations, resize_option, custom_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "Savanthgc/MyCustomDiffusionModel",## api Hashcode  change this to custom dataset link if requried.
            revision="v2.0",
        ).to(device)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    try:
        im = Image.open(image_path)
    except FileNotFoundError:
        print("The specified image file was not found.")
        return

    if resize_option == 'custom' and custom_size:
        target_size = custom_size
    else:
        target_size = (224, 224)  # Default size

    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            target_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(im).to(device).unsqueeze(0)

    out = sd_pipe(inp, num_inference_steps=50, guidance_scale=7.5)
    for i, img in enumerate(out["images"][:num_variations]):
        img.save(f"outputs/result_{i+1}.jpg")
        print(f"Variation {i+1} saved.")
