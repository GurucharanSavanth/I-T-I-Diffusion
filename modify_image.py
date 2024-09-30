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
from torchvision.utils import save_image
import os
import nltk
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import psutil
import numpy as np

nltk.download('punkt')

def check_system_resources():
    """Check available memory and CPU usage to decide processing strategy."""
    memory_info = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"Available Memory: {memory_info.available / (1024 * 1024)} MB")
    print(f"CPU Usage: {cpu_usage}%")

    if memory_info.available < (2 * 1024 * 1024 * 1024):  # Less than 2 GB
        print("Low memory: Reducing batch size or inference steps.")
        return False
    return True

def dynamic_adjustment_based_on_prompt(prompt):
    tokens = nltk.word_tokenize(prompt)
    word_count = len(tokens)
    if word_count < 5:
        return 5, 7.0
    else:
        return 50, 10.0

def classify_image(image_tensor):
    flattened_image = image_tensor.view(-1).cpu().numpy()
    knn = KNeighborsClassifier(n_neighbors=3)
    xgb = XGBClassifier()

    X = np.random.rand(100, len(flattened_image))
    y = np.random.randint(0, 2, 100)
    knn.fit(X, y)
    xgb.fit(X, y)

    knn_class = knn.predict([flattened_image])
    xgb_class = xgb.predict([flattened_image])

    print(f"KNN Classification: {knn_class}, XGBoost Classification: {xgb_class}")
    return knn_class, xgb_class

def modify_image(image_path, prompt, strength, num_variations, resize_option, custom_size, batch_size=1, apply_custom_transforms=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not check_system_resources():
        batch_size = max(1, batch_size // 2)
        num_variations = max(1, num_variations // 2)
        print(f"Adjusted batch size: {batch_size}, num_variations: {num_variations}")

    try:
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="main",
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
        target_size = (224, 224)

    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])

    inp = tform(im).to(device).unsqueeze(0)

    classify_image(inp)

    if apply_custom_transforms:
        inp = transforms.RandomHorizontalFlip()(inp)

    all_images = []
    for _ in range(batch_size):
        inference_steps, guidance = dynamic_adjustment_based_on_prompt(prompt)
        out = sd_pipe(inp, num_inference_steps=inference_steps, guidance_scale=guidance)
        all_images.extend(out["images"][:num_variations])

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    for i, img in enumerate(all_images):
        img.save(f"outputs/result_{i + 1}.jpg")
        print(f"Variation {i + 1} saved.")

    if len(all_images) > 1:
        all_tensors = [transforms.ToTensor()(img) for img in all_images]
        save_image(torch.stack(all_tensors), 'outputs/collage.jpg', nrow=5)
        print("Collage of all variations saved.")

    return len(all_images)
