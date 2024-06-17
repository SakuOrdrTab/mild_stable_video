import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, StableDiffusionPipeline
from PIL import Image

# Init pipeline, use cuda
model_name = "stabilityai/stable-diffusion-2-base"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, safety_checker=None).to("cuda")
output_pipe = StableDiffusionPipeline.from_pretrained(model_name, safety_checker=None).to("cuda")

init_image = Image.open("sample_image.jpg").convert("RGB")

# The prompt
prompt = "A finnish marsh at night"

# Disable gradient computation for inference
with torch.no_grad():
    # Generate latent representations
    latent_result = pipe(prompt=prompt,
                         image=init_image,
                         strength=0.35,
                         guidance_scale=10,
                         output_type="latent")

    # Extract the latent representation
    latents = latent_result.images

    # latents 

    # Decode the latents to an image
    latent_image_tensor = pipe.vae.decode(latents).sample
    latent_image_tensor = (latent_image_tensor / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]

    # Convert the tensor to numpy
    latent_image_numpy = latent_image_tensor[0].permute(1, 2, 0).cpu().numpy()

    # Convert to PIL Image
    latent_image = Image.fromarray((latent_image_numpy * 255).astype(np.uint8))

    # Generate final image from latent representation
    final_image = output_pipe(prompt=prompt,
                              latents=latents,
                              num_inference_steps=30,
                              guidance_scale=10,
                              output_type="pil").images[0]

# Combine the images for display
width, height = init_image.size
combined_image = Image.new('RGB', (width * 3, height))

combined_image.paste(init_image, (0, 0))
combined_image.paste(latent_image, (width, 0))
combined_image.paste(final_image, (width * 2, 0))

combined_image.show()
