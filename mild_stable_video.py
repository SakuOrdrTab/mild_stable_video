import numpy as np
import subprocess
import os

from PIL import Image

from mildlyStableVideoPipeline import MildlyStableVideoPipeline
from mildlyStableXLVideoPipeline import MildlyStableXLVideoPipeline
from latentStableVideoPipeline import LatentStableVideoPipeline
from opticalFlowStableVideoPipeline import OpticalFlowStableVideoPipeline
from mildlyStableDepthVideoPipeline import MildlyStableDepthVideoPipeline
from flowWarpedVideoPipeline import FlowWarpedVideoPipeline

# Constants
SD_MARKER = "_FW" # How to mark what actual pipeline was used for the output video name
INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_videos"
INPUT_VIDEO_NAME = "tyttospider_tekee_paketin.mp4"
OUTPUT_VIDEO_NAME = INPUT_VIDEO_NAME.split(".")[0] + SD_MARKER + ".mpeg"

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
    input_video = os.path.join(INPUT_DIR, INPUT_VIDEO_NAME)
    output_video = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_NAME)

    prompt =  "Terrible monster spiders in a web, cinematic lighting, psychedelic colors, nightmarish"
    negative_prompt = "distortions, blurred"

    # Transform the video
    # NOTE: needs more continuity
    pipe = FlowWarpedVideoPipeline()
    pipe.do_magic(input_video, output_video,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=13,
                image_strength=0.25,
                inferring_steps=30, 
                initial_frame_guidance_scale=13, 
                initial_image_strength=0.25, 
                last_frame_weight=0.4)
    
    # At least on Windows, the output video does not necessarily work on default media player (works with ffplay, though)
    # Convert the video to mp4 to ensure compatibility
    convert_video_to_mp4(output_video, output_video.split(".")[0] + ".mp4")

