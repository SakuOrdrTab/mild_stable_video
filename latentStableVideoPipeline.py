import numpy as np

import cv2

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler

from PIL import Image

from mildlyStableVideoPipeline import MildlyStableVideoPipeline

# How many imagesd are saved in an array and blended
LATENT_ARRAY_SIZE = 3


class LatentStableVideoPipeline(MildlyStableVideoPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name += " + Latent blending"
        self._earlier_latents = []

    def _blend_latents(self):
        # Stack the arrays along a new axis (axis=0)
        stacked_arrays = np.stack(self._earlier_latents, axis=0)

        # Calculate the average of the elements along the new axis (axis=0)
        averaged_array = np.mean(stacked_arrays, axis=0)

        return averaged_array

    def _transform_frame(self, frame):
        # Resize the frame to 640x480
        resized_frame = cv2.resize(frame, (640, 480))

        # Convert resized frame to PIL image for pipeline
        pil_image = Image.fromarray(resized_frame)

        with torch.no_grad():
            # Generate latent representations
            latent_result = self._pipe(
                prompt=self._prompt,
                image=pil_image,  # ensure you use pil_image instead of frame
                strength=0.05,
                guidance_scale=10,
                output_type="latent",
            )

        latent_result = latent_result.images

        latent_image_tensor = self._pipe.vae.decode(
            latent_result
        ).sample  # Get the tensor from DecoderOutput

        # Rescale the tensor values from [-1, 1] to [0, 1]
        latent_image_tensor = (latent_image_tensor + 0.5).clamp(0, 1)

        from torchvision.transforms.functional import to_pil_image

        # Assuming latent_image_tensor is already in the range [0, 1]
        # You can create a PIL image directly from a PyTorch tensor on GPU
        latent_image = to_pil_image(
            latent_image_tensor[0]
        )  # Convert the first tensor in the batch to a PIL image

        # Manage the list of earlier latents
        if len(self._earlier_latents) < LATENT_ARRAY_SIZE:
            self._earlier_latents.append(latent_image)
        else:
            # Remove the oldest latent and append the new one
            self._earlier_latents.pop(0)
            self._earlier_latents.append(latent_image)

        # Blend latent vectors
        blended_latent = self._blend_latents()

        # Generate final image from latent representation
        final_image = self._pipe(
            prompt=self._prompt,
            negative_prompt=self._negative_prompt,
            image=blended_latent,
            strength=self._strength,
            guidance_scale=self._guidance,
            num_inference_steps=self._passes,
            output_type="pil",
        ).images[0]

        # Convert decoded image to numpy array
        transformed_frame = np.array(final_image)

        # Update the last transformed image (for other processing)
        self._last_transformed_image = transformed_frame.copy()

        return transformed_frame


if __name__ == "__main__":
    test_pipe = LatentStableVideoPipeline()
    # Force some attributes to parent class to test function
    test_pipe._prompt = "japanese wood painting"
    test_pipe._negative_prompt = ""
    test_pipe._initial_frame_guidance_scale = 15
    test_pipe._initial_image_strength = 0.7

    # Load sample initial image
    init_image = np.asarray(Image.open("sample_image.jpg"))
    # Show result that the pipeline returns as an array
    Image.fromarray(test_pipe._first_frame(init_image)).show()
