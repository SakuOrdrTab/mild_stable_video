import numpy as np
from PIL import Image

from mildlyStableVideoPipeline import MildlyStableVideoPipeline

from diffusers import StableDiffusionDepth2ImgPipeline

import torch

class MildlyStableDepthVideoPipeline(MildlyStableVideoPipeline):
    """Stable Diffusion depth model pipeline"""
    def __init__(self):
        self._model_name = "stabilityai/stable-diffusion-2-depth"
        self._pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(self._model_name,
                                                               torch_dtype=torch.float16, 
                                                               variant="fp16", 
                                                               use_safetensors=True,
                                                               safety_checker=None
                                                            ).to("cuda")
        self._last_transformed_image = None

    def _first_frame(self, frame):
        """The first frame of the sequence gets a special treatment. There is no previous
        transformed image, and the frame is not blended. SD XL seems to require more passes, too

        Args:
            frame (imageio frame): The first frame of the video.

        Returns:
            numpy array (2D): The tranformed frame
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
    
if __name__ == "__main__":
    test_pipe = MildlyStableDepthVideoPipeline()
    # Force some attributes to parent class to test function
    test_pipe._prompt = "japanese wood painting"
    test_pipe._negative_prompt = ""
    test_pipe._initial_frame_guidance_scale = 15
    test_pipe._initial_image_strength = 0.7

    # Load sample initial image
    init_image = np.asarray(Image.open("sample_image.jpg"))
    # Show result that the pipeline returns as an array
    Image.fromarray(test_pipe._first_frame(init_image)).show()