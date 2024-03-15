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

from mildlyStableVideoPipeline import MildlyStableVideoPipeline
from mildlyStableXLVideoPipeline import MildlyStableXLVideoPipeline

def convert_video_to_mp4(input_video, output_video):
    """Converts the video to mp4 format using ffmpeg. Uses external subprocess to run the command.

    Args:
        input_video (str): The video file path to be converted.
        output_video (str): the output video file path.
    """    
    command = [
        'ffmpeg',
        '-i', input_video,
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '22',  # Lower numbers are higher quality, 18-28 is a good range
        '-c:a', 'aac',  # Convert audio to AAC for wide compatibility
        '-strict',  # Necessary if using older versions of FFmpeg for AAC
        'experimental',
        output_video
    ]
    try:
        subprocess.run(command)
    except Exception as e:
        print(f"Error converting video: {e}")


if __name__ == "__main__":
    input_video = "sample1.mpeg"
    output_video = input_video.split(".")[0] + "_SD" + ".mpeg"

    prompt = "Hallucinogenic colours, with glowing mushrooms illuminating the path for tiny glowing creatures. Strange, otherworldly plants grow amongst the reeds"
    negative_prompt = "realistic, ordinary, daytime"

    # Transform the video
    pipe = MildlyStableXLVideoPipeline()
    pipe.do_magic(input_video, output_video,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=15,
                image_strength=0.3,
                inferring_steps=15, 
                initial_frame_guidance_scale=19, 
                initial_image_strength=0.3, 
                last_frame_weight=0.5)
    
    # At least on Windows, the output video does not necessarily work on default media player (works with ffplay, though)
    # Convert the video to mp4 to ensure compatibility
    convert_video_to_mp4(output_video, output_video.split(".")[0] + ".mp4")

