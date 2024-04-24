import numpy as np

# https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
import cv2

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler

from PIL import Image

from mildlyStableVideoPipeline import MildlyStableVideoPipeline

LATENT_ARRAY_SIZE = 3

class OpticalFlowStableVideoPipeline(MildlyStableVideoPipeline):

    def __init__(self):
        super().__init__()

    def _apply_optical_flow(self, prev_frame, current_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Create a meshgrid of pixel coordinates
        h, w = current_gray.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply the flow displacements to the original pixel positions
        displacement_x = x + flow[..., 0]
        displacement_y = y + flow[..., 1]

        # Use remap with the correct displacement values
        warped_frame = cv2.remap(current_frame, displacement_x.astype(np.float32), displacement_y.astype(np.float32), cv2.INTER_LINEAR)
        
        return warped_frame
    
    def _transform_frame(self, frame):
        # Resize the frame to 640x480
        resized_frame = cv2.resize(frame, (640, 480))

        # Use optical flow if there's a last transformed image
        if self._last_transformed_image is not None:
            frame_to_process = self._apply_optical_flow(self._last_transformed_image, resized_frame)
            # You can still blend the frames after applying optical flow for smoother transitions
            frame_to_process = self._blend_images(last_image=self._last_transformed_image, new_frame=frame_to_process)
        else:
            frame_to_process = resized_frame
            # Use the first frame processing and return the result
            return self._first_frame(frame_to_process)
        
        pil_image = Image.fromarray(frame_to_process)

        # Process the PIL image with the pipeline
        with torch.no_grad():
            transformed_image = self._pipe(prompt=self._prompt,
                                           negative_prompt=self._negative_prompt,
                                           image=pil_image,
                                           strength=self._strength,
                                           guidance_scale=self._guidance,
                                           num_inference_steps=self._passes).images[0]

        # Convert to numpy array to be processed as a frame
        transformed_frame = np.array(transformed_image)

        # Update the last transformed image
        self._last_transformed_image = transformed_frame.copy()

        return transformed_frame
    