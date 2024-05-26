import numpy as np
import os
import json
import subprocess
import time
import shutil

import imageio # pip install imageio imageio-ffmpeg
import cv2

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler

from PIL import Image


model_name = "stabilityai/stable-diffusion-2-base"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, safety_checker=None)

# Use cuda (graphics processor) if possible, If not, you probably have to use different arguments for  pipe init, too:
# revision="fp16"
# torch_dtype=torch.float16
pipe = pipe.to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

pil_image = Image.open("sample_image.jpg")

prompt = "japanese wood painting"

with torch.no_grad():  # Use torch.no_grad() to reduce memory usage
    latents = pipe(prompt=prompt,
                                    image=pil_image,
                                    strength=0.3,
                                    guidance_scale=10,
                                    output_type="latent").images
    image = pipe.decode_latents(latents)
    transformed_image = pipe.numpy_to_pil(image)[0]
    

    
transformed_image.show()