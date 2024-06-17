import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, StableDiffusionPipeline
from PIL import Image

# CHECK:
# https://github.com/huggingface/diffusers/issues/2871

# Init pipeline, use cuda
model_name = "stabilityai/stable-diffusion-2-base"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, safety_checker=None).to("cuda")
output_pipe = StableDiffusionPipeline.from_pretrained(model_name, safety_checker=None).to("cuda")

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
    latent_image_tensor = pipe.vae.decode(latents).sample  / 0.18215  # Get the tensor from DecoderOutput

    print("Latent tensor min:", latent_image_tensor.min())
    print("Latent tensor max:", latent_image_tensor.max())

    # Convert the tensor to CPU and change dimensions for image conversion
    #   latent_image_tensor = latent_image_tensor.cpu() + 0.5

    # Remove extra dimensions and ensure the tensor is in the correct format (H, W, C)
    latent_image_numpy = latents[0].permute(1, 2, 0).cpu().numpy()  # Assuming 'latent' is the batch with one image

    latent_image_numpy = latent_image_numpy  +  0.5

    # Ensure the data is in the correct range and type
    latent_image_numpy = (latent_image_numpy * 255).clip(0, 255).astype(np.uint8)

    # Convert to PIL Image
    latent_image = Image.fromarray(latent_image_numpy)

    # Generate final image from latent representation
    final_image = output_pipe(prompt=prompt,

                       latents=latents,
                       strength=0.15,
                       guidance_scale=10,
                       output_type="pil").images[0]
    
width, height = init_image.size
combined_image = Image.new('RGB', (width * 3, height))

combined_image.paste(init_image, (0, 0))
combined_image.paste(latent_image, (width, 0))
combined_image.paste(final_image, (width * 2, 0))

combined_image.show()
