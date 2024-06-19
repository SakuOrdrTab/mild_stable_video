import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

model_name = "stabilityai/stable-diffusion-2-base"
input_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, safety_checker=None).to("cuda")
output_pipe = StableDiffusionPipeline.from_pretrained(model_name, safety_checker=None).to("cuda")

init_image = Image.open("sample_image.jpg").convert("RGB")

prompt = "A Japanese wood painting"

with torch.no_grad():
    initial_image = input_pipe(prompt=prompt, # Get img2img SD image
                         image=init_image,
                         strength=0.35,
                         guidance_scale=10,
                         output_type="pil").images[0]

    # Convert PIL image to tensor and normalize to [-1, 1]
    initial_image_tensor = torch.tensor(np.array(initial_image)).permute(2, 0, 1)\
                                .unsqueeze(0).float().to("cuda") / 255.0
    initial_image_tensor = (initial_image_tensor - 0.5) * 2  

    # Encode image to latent space
    latents = input_pipe.vae.encode(initial_image_tensor).latent_dist.sample() * 0.18215

    # Decode back latents to an image for visualization
    decoded_latent_tensor = input_pipe.vae.decode(latents)['sample']
    decoded_latent_tensor = (decoded_latent_tensor / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
    decoded_latent_numpy = decoded_latent_tensor[0].permute(1, 2, 0).cpu().numpy()
    decoded_latent_pil = Image.fromarray((decoded_latent_numpy * 255).astype(np.uint8))

    # Generate final image from latent representation
    final_image = output_pipe(prompt=prompt,
                              latents=latents,
                              num_inference_steps=50,  # Increase steps for better quality
                              guidance_scale=7.5,  # Adjust guidance scale for better prompt adherence
                              output_type="pil").images[0]

# Show the results: initial image, SD img2img, latent representation image, SD from latent
width, height = init_image.size
combined_image = Image.new('RGB', (width * 4, height))
combined_image.paste(init_image, (0, 0))
combined_image.paste(initial_image, (width, 0))
combined_image.paste(decoded_latent_pil, (width * 2, 0))
combined_image.paste(final_image, (width * 3, 0))

combined_image.show()