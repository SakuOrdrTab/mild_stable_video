import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from PIL import Image

# CHECK:
# https://github.com/huggingface/diffusers/issues/2871

# Init pipeline, use cuda
model_name = "stabilityai/stable-diffusion-2-base"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, safety_checker=None).to("cuda")

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

    print("Latent tensor min:", latent_image_tensor.min())
    print("Latent tensor max:", latent_image_tensor.max())

    # Rescale the tensor values from [-1, 1] to [0, 1]
    latent_image_tensor = (latent_image_tensor + 0.5).clamp(0, 1)

    # Convert the tensor to CPU and change dimensions for image conversion
    latent_image_numpy = latent_image_tensor.cpu().permute(0, 2, 3, 1).numpy()

    # Extract channels
    r_channel = latent_image_numpy[0, :, :, 0]
    g_channel = latent_image_numpy[0, :, :, 1]
    b_channel = latent_image_numpy[0, :, :, 2]

    # Analyze the mean values of each channel
    print("Red channel mean:", r_channel.mean())
    print("Green channel mean:", g_channel.mean())
    print("Blue channel mean:", b_channel.mean())

    # Adjust the blue channel if significantly different
    if b_channel.mean() < 0.4:  # Check if the blue channel is notably lower
        print("Changing blue channel!")
        b_channel = b_channel * 1.5  # Scale up blue channel values
        b_channel = np.clip(b_channel, 0, 1)  # Ensure values stay within [0, 1]
        r_channel = b_channel * 0.9  # Scale up blue channel values
        r_channel = np.clip(b_channel, 0, 1)  # Ensure values stay within [0, 1]
        g_channel = b_channel * 1.0  # Scale up blue channel values
        g_channel = np.clip(b_channel, 0, 1)  # Ensure values stay within [0, 1]


    # Recombine channels
    adjusted_image_numpy = np.stack([r_channel, g_channel, b_channel], axis=-1)

    # Show result of colour adjustment
    print("Channel means:", adjusted_image_numpy[:, :, 0].mean(), adjusted_image_numpy[:, :, 1].mean(), adjusted_image_numpy[:, :, 2].mean())

    # Convert back to PIL image using the adjusted array
    latent_image = Image.fromarray((adjusted_image_numpy * 255).astype(np.uint8))

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
