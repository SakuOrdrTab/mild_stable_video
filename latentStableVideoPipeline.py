import numpy as np

import cv2

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler

from PIL import Image

from mildlyStableVideoPipeline import MildlyStableVideoPipeline

LATENT_ARRAY_SIZE = 3

class LatentStableVideoPipeline(MildlyStableVideoPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name += " + Latent blending"
        self._earlier_latents = [None] * LATENT_ARRAY_SIZE

    def _blend_latents(self, current_latent):
        # Simple blending with weights (adjust as needed)
        blended_latent = 0.15 * self._earlier_latents[0] + 0.7 * current_latent + 0.15 * self._earlier_latents[-2]
        return blended_latent
    
    def _transform_frame(self, frame):
        # Resize the frame to 640x480
        resized_frame = cv2.resize(frame, (640, 480))

        # Convert resized frame to PIL image for pipeline
        pil_image = Image.fromarray(resized_frame)

        with torch.no_grad():  # Use torch.no_grad() to reduce memory usage
            # Process the PIL image with the pipeline, returning the latent
            diffusion_output = self._pipe(prompt=self._prompt,
                                            negative_prompt=self._negative_prompt,
                                            image=pil_image,
                                            strength=self._strength,
                                            guidance_scale=self._guidance,
                                            num_inference_steps=self._passes,
                                            return_latent=True)

        current_latent = diffusion_output["latent"]

        # Blend latent vectors
        blended_latent = self._blend_latents(current_latent)

        # Decode blended latent back to image
        decoded_image = self._pipe(latent=blended_latent, return_pil=True)["sample"]

        # Convert decoded image to numpy array
        transformed_frame = np.array(decoded_image)

        # Update the latent array with the current latent
        self._earlier_latents.append(current_latent)
        self._earlier_latents = self._earlier_latents[-LATENT_ARRAY_SIZE:]

        # Update the last transformed image (for other processing)
        self._last_transformed_image = transformed_frame.copy()

        return transformed_frame
    