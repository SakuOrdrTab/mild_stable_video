import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

# Initialize pipelines, use CUDA
model_name = "stabilityai/stable-diffusion-2-base"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, safety_checker=None).to("cuda")
output_pipe = StableDiffusionPipeline.from_pretrained(model_name, safety_checker=None).to("cuda")

init_image = Image.open("sample_image.jpg").convert("RGB")

# The prompt
prompt = "A Japanese wood painting"

# Disable gradient computation for inference
with torch.no_grad():
    # Generate initial image
    latent_result = pipe(prompt=prompt,
                         image=init_image,
                         strength=0.35,
                         guidance_scale=10,
                         output_type="pil")

    # Extract the image
    latent_image = latent_result.images[0]

    # Convert PIL image to latent space
    latent_image_tensor = torch.tensor(np.array(latent_image)).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 255.0
    latent_image_tensor = (latent_image_tensor - 0.5) * 2  # Normalize to [-1, 1]

    # Encode image to latent space
    latents = pipe.vae.encode(latent_image_tensor).latent_dist.sample() * 0.18215

    # Decode the latents to an image for visualization
    decoded_image_tensor = pipe.vae.decode(latents)['sample']
    decoded_image_tensor = (decoded_image_tensor / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]

    # Convert the tensor to numpy
    decoded_image_numpy = decoded_image_tensor[0].permute(1, 2, 0).cpu().numpy()

    # Convert to PIL Image
    decoded_image_pil = Image.fromarray((decoded_image_numpy * 255).astype(np.uint8))

    # Generate final image from latent representation
    generator = torch.manual_seed(42)  # Ensure deterministic results

    final_image = output_pipe(prompt=prompt,
                              latents=latents,
                              num_inference_steps=50,  # Increase steps for better quality
                              guidance_scale=7.5,  # Adjust guidance scale for better prompt adherence
                              generator=generator,
                              output_type="pil").images[0]

# Combine the images for display
width, height = init_image.size
combined_image = Image.new('RGB', (width * 4, height))

combined_image.paste(init_image, (0, 0))
combined_image.paste(latent_image, (width, 0))
combined_image.paste(decoded_image_pil, (width * 2, 0))
combined_image.paste(final_image, (width * 3, 0))

combined_image.show()



