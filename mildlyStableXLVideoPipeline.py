import numpy as np
from PIL import Image

from mildlyStableVideoPipeline import MildlyStableVideoPipeline

from diffusers import AutoPipelineForImage2Image

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import DDIMScheduler
import torch

class MildlyStableXLVideoPipeline(MildlyStableVideoPipeline):
    """Stable Diffusion XL model pipeline"""
    def __init__(self):
        self._model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        self._txt2img_pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                               torch_dtype=torch.float16, 
                                                               variant="fp16", 
                                                               use_safetensors=True,
                                                               safety_checker=None
                                                            ).to("cuda")
        self._pipe = AutoPipelineForImage2Image.from_pipe(self._txt2img_pipeline).to("cuda")

        self._pipe.scheduler = DDIMScheduler.from_config(self._pipe.scheduler.config)
        self.scheduler_name = self._pipe.scheduler.config._class_name

        self._last_transformed_image = None

    def _first_frame(self, frame):
        """The first frame of the sequence gets a special treatment. There is no previous
        transformed image, and the frame is not blended. SD XL seems to require more passes, too

        Args:
            frame (imageio frame): The first frame of the video.

        Returns:
            PIL Image: The tranformed frame
        """        
        pil_image = Image.fromarray(frame)

        with torch.no_grad():  # Use torch.no_grad() to reduce memory usage
            transformed_image = self._pipe(prompt=self._prompt,
                                            image=pil_image,
                                            num_inference_steps=40,
                                            strength=self._initial_image_strength,
                                            guidance_scale=self._initial_frame_guidance_scale).images[0]

        transformed_frame = np.array(transformed_image)
        self._last_transformed_image = transformed_frame.copy()
        return transformed_frame