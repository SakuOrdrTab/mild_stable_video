import numpy as np


import torch
from diffusers import AutoPipelineForImage2Image

from PIL import Image

from mildlyStableVideoPipeline import MildlyStableVideoPipeline 


class MildlyKandinsky3VideoPipeline(MildlyStableVideoPipeline):
    def __init__(self, ):
        self._pipe = AutoPipelineForImage2Image.from_pretrained("kandinsky-community/kandinsky-3", 
                                                                variant="fp16",
                                                                torch_dtype=torch.float16).to("cuda")
        self._model_name = "kandinsky-3"
        self._last_transformed_image = None


if __name__ == "__main__":
    test_pipe = MildlyKandinsky3VideoPipeline()
    # Force some attributes to parent class to test function
    test_pipe._prompt = "japanese wood painting"
    test_pipe._negative_prompt = ""
    test_pipe._initial_frame_guidance_scale = 15
    test_pipe._initial_image_strength = 0.7

    # Load sample initial image
    init_image = np.asarray(Image.open("sample_image.jpg"))
    # Show result that the pipeline returns as an array
    Image.fromarray(test_pipe._first_frame(init_image)).show()