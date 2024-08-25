import torch
from PIL import Image
from diffusers import StableDiffusionImageVariationPipeline
from torchvision import transforms

def modify_image(image_path, prompt, strength, num_variations, resize_option, custom_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "U2FsdGVkX1/wqzTdNhIIf1sAo/d7F/f3dgNgnNwVMLyvU7yQHVwM2vDjJ7u4dv52r84otG9UzunLQPdQKr7R+A==",## api Hashcode  change this to custom dataset link if requried.
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
