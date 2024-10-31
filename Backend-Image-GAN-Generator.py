import os
import requests
import time
import random
import numpy as np
import concurrent.futures
from threading import Lock
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionImageVariationPipeline, DDIMScheduler
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from torchvision import transforms
import xgboost as xgb
from torch import nn

# Class for handling all folder setup
class FolderManager:
    @staticmethod
    def setup_folders():
        folders = ["images_scraped", "generated_images", "fine_tuned_models"]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

# Class for Image Scraping
class ImageScraper:
    def __init__(self, query, num_images=16):
        self.query = query
        self.num_images = num_images
        self.image_urls = []

    def scrape_images(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={self.query}"
        driver.get(search_url)
        time.sleep(2)

        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        img_elements = driver.find_elements(By.CSS_SELECTOR, "img.rg_i.Q4LuWd")
        for img in img_elements:
            src = img.get_attribute("src")
            if src and src.startswith("http"):
                self.image_urls.append(src)
            if len(self.image_urls) >= self.num_images:
                break
        
        driver.quit()
        return self.image_urls

# Class for Image Downloading
class ImageDownloader:
    def __init__(self, image_urls):
        self.image_urls = image_urls
        self.lock = Lock()

    def download_images(self):
        def download_image(idx, url):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                image_path = f"images_scraped/image_{idx}.jpg"
                with open(image_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            except Exception as e:
                with self.lock:
                    print(f"Failed to download image {idx}: {e}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(download_image, idx, url) for idx, url in enumerate(self.image_urls)]
            concurrent.futures.wait(futures)

# Class for Dataset Preparation
class DatasetPreparer:
    @staticmethod
    def prepare_dataset(image_dir="images_scraped"):
        dataset = []
        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            try:
                image = Image.open(img_path).resize((64, 64))
                image = np.array(image) / 127.5 - 1.0  # Normalize to [-1, 1]
                if image.shape == (64, 64, 3):
                    dataset.append(image)
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
        if len(dataset) == 0:
            print("No valid images found. Generating random placeholder images for training.")
            dataset = np.random.normal(0, 1, (16, 64, 64, 3))
        return np.array(dataset)

# Class for Generator and Discriminator
class GANComponents:
    class Generator(nn.Module):
        def __init__(self):
            super(GANComponents.Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 3 * 64 * 64),
                nn.Tanh()
            )

        def forward(self, x):
            return self.model(x).view(-1, 3, 64, 64)

    class Discriminator(nn.Module):
        def __init__(self):
            super(GANComponents.Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(3 * 64 * 64, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x.view(-1, 3 * 64 * 64))

# Class for handling Model Generation and Uploading
class ModelGeneratedUploader:
    def __init__(self):
        self.model_name = "Savanthgc/MyCustomDiffusionModel"
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            self.model_name,
            revision="main"
        ).to("cuda")
        self.generator = GANComponents.Generator().to("cuda")
        self.discriminator = GANComponents.Discriminator().to("cuda")

    def fine_tune_model(self, dataset, epochs=5):
        print("Starting model adjustment with the provided dataset...")
        optimizer_pipe = torch.optim.Adam(self.pipe.unet.parameters(), lr=1e-4)
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            for i, img in enumerate(dataset):
                # Fine-tuning diffusion model
                noise = torch.randn((1, 4, 64, 64), device="cuda")
                latents = self.pipe.vae.encode(torch.from_numpy(img).to("cuda")).latent_dist.sample()
                latents = latents * 0.18215
                noisy_latents = latents + noise
                loss_pipe = torch.mean((noisy_latents - latents) ** 2)
                optimizer_pipe.zero_grad()
                loss_pipe.backward()
                optimizer_pipe.step()

                # GAN training
                real_imgs = torch.from_numpy(img).float().to("cuda")
                valid = torch.ones((1, 1), device="cuda")
                fake = torch.zeros((1, 1), device="cuda")

                # Train Generator
                z = torch.randn((1, 100), device="cuda")
                generated_imgs = self.generator(z)
                g_loss = criterion(self.discriminator(generated_imgs), valid)
                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()

                # Train Discriminator
                real_loss = criterion(self.discriminator(real_imgs), valid)
                fake_loss = criterion(self.discriminator(generated_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()

                print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(dataset)}, Loss Pipe: {loss_pipe.item():.4f}, Loss G: {g_loss.item():.4f}, Loss D: {d_loss.item():.4f}")
        print("Model adjustment complete.")

    def generate_images(self, prompt, num_images=4):
        for i in range(num_images):
            image = self.pipe(prompt).images[0]
            image.save(f"generated_images/generated_{i}.png")

# Class for managing the entire Image Pipeline
class ImagePipeline:
    def __init__(self, query):
        self.query = query
        FolderManager.setup_folders()

    def execute(self):
        # Step 1: Scrape Images
        scraper = ImageScraper(self.query)
        image_urls = scraper.scrape_images()

        # Step 2: Download Images
        downloader = ImageDownloader(image_urls)
        downloader.download_images()

        # Step 3: Prepare Dataset
        dataset = DatasetPreparer.prepare_dataset()

        # Step 4: Fine-Tune Model
        model_handler = ModelGeneratedUploader()
        model_handler.fine_tune_model(dataset, epochs=5)

        # Step 5: Generate Images
        model_handler.generate_images(self.query, num_images=4)

if __name__ == "__main__":
    user_query = input("Enter an image prompt: ")
    pipeline = ImagePipeline(user_query)
    pipeline.execute()
