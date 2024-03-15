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