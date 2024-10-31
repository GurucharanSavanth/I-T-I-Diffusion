import torch
from diffusers import StableDiffusionPipeline

# Load the custom diffusion model from Hugging Face
model_name = "Savanthgc/MyCustomDiffusionModel"
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

# Define the prompt for image generation
prompt = "A beautiful landscape with mountains and rivers during sunset"

# Generate the image
with torch.autocast("cuda"):
    image = pipe(prompt).images[0]

# Display the image
image.show()

# Optionally, save the image
image.save("generated_image.png")
