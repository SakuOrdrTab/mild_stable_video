import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from PIL import Image

# Init pipeline, use cuda
model_name = "stabilityai/stable-diffusion-2-base"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, safety_checker=None).to("cuda")

# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

init_image = Image.open("sample_image.jpg")

# The prompt
prompt = "japanese wood painting"

# Disable gradient computation for inference
with torch.no_grad():
    # Generate latent representations
    latent_result = pipe(prompt=prompt,
                         image=init_image,
                         strength=0.05,
                         guidance_scale=10,
                         output_type="latent")

    # Extract the latent representation
    latents = latent_result.images

    # Have to use Vae as pipe.decode_latents is deprecated :(
    latent_image_tensor = pipe.vae.decode(latents).sample  # Get the tensor from DecoderOutput

    # Rescale the tensor values from [-1, 1] to [0, 1]
    latent_image_tensor = (latent_image_tensor / 2 + 0.5).clamp(0, 1)

    # Move the tensor to the CPU and change the dimensions to (height, width, channels)
    latent_image_numpy = latent_image_tensor.cpu().permute(0, 2, 3, 1).numpy()

    # Convert to PIL image
    latent_image = Image.fromarray((latent_image_numpy[0] * 255).astype(np.uint8))

    # Generate final image from latent representation
    final_image = pipe(prompt=prompt,
                       image=latent_image,
                       strength=0.3,
                       guidance_scale=10,
                       output_type="pil").images[0]
    
width, height = init_image.size
combined_image = Image.new('RGB', (width * 3, height))

combined_image.paste(init_image, (0, 0))
combined_image.paste(latent_image, (width, 0))
combined_image.paste(final_image, (width * 2, 0))

combined_image.show()