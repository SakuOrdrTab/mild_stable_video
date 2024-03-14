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

class MildlyStableVideoPipeline():
    def __init__(self, video_name: str, output_name : str, initial_frame_guidance_scale : int =15):
        self._video_name = video_name
        self._output_name = output_name
        self._frames_folder = os.path.join(".", "temp")

        self._model_name = "runwayml/stable-diffusion-v1-5"
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_name, safety_checker=None)

        # Use cuda (graphics processor) if possible, If not, you probably have to use different arguments for  pipe init, too:
        # revision="fp16"
        # torch_dtype=torch.float16
        self._pipe = self._pipe.to("cuda")

        self._pipe.scheduler = DDIMScheduler.from_config(self._pipe.scheduler.config)
        self.scheduler_name = self._pipe.scheduler.config._class_name

        self._prompt = "" # Default prompt

        self._guidance = 20
        self._strength = 0.15
        self._passes = 10

        self._last_transformed_image = None  # Initialize with None
        self._initial_frame_guidance_scale = initial_frame_guidance_scale

    @property
    def model_name(self):
        return self._model_name 
    
    def _first_frame(self, frame):
        """The first frame of the sequence gets a special treatment. There is no previous
        transformed image, and the frame is not blended. Also the prompt gets more guidance.

        Args:
            frame (imageio frame): The first frame of the video.

        Returns:
            PIL Image: The tranformed frame
        """        
        pil_image = Image.fromarray(frame)

        with torch.no_grad():  # Use torch.no_grad() to reduce memory usage
            transformed_image = self._pipe(prompt=self._prompt,
                                            image=pil_image,
                                            strength=self._strength,
                                            guidance_scale=15).images[0]

        transformed_frame = np.array(transformed_image)
        self._last_transformed_image = transformed_frame.copy()
        return transformed_frame

    def _transform_frame(self, frame):
        # Resize the frame to 640x480
        resized_frame = cv2.resize(frame, (640, 480))

        # If there's a last transformed image, blend it with the current frame
        if self._last_transformed_image is not None:
            # Ensure last transformed image is resized to match the current frame
            last_image_resized = cv2.resize(self._last_transformed_image, (640, 480))
            frame_to_process = self._blend_images(resized_frame, last_image_resized)
        else:
            # return a more processed image
            frame_to_process = resized_frame
            return self._first_frame(frame_to_process)
        
        pil_image = Image.fromarray(frame_to_process)

        # Process the PIL image with the pipeline
        with torch.no_grad():  # Use torch.no_grad() to reduce memory usage
            transformed_image = self._pipe(prompt=self._prompt,
                                            image=pil_image,
                                            strength=self._strength,
                                            guidance_scale=self._guidance).images[0]

        # Convert to numpy array to be processed as a frame
        transformed_frame = np.array(transformed_image)

        # Update the last transformed image
        self._last_transformed_image = transformed_frame.copy()

        return transformed_frame

    def _blend_images(self, image1, image2):
        '''Blends the last frame (tranformed) with the current frame'''
        return cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
    
    def _extract_and_transform_frames(self):
        # Ensure frames folder exists
        if not os.path.exists(self._frames_folder):
            os.makedirs(self._frames_folder)

        # Create a reader object for the video
        reader = imageio.get_reader(self._video_name)
        metadata = reader.get_meta_data()

        # Iterate over, transform, and save each frame as an image
        for i, frame in enumerate(reader):
            transformed_frame = self._transform_frame(frame)
            image_path = os.path.join(self._frames_folder, f"frame_{i:04d}.png")
            if not os.path.exists(image_path):  # Only process and save if the frame hasn't been saved before
                imageio.imwrite(image_path, transformed_frame)
                print(f"Processed and saved frame {i}")

        # Save metadata to a JSON file in the frames folder
        metadata_path = os.path.join(self._frames_folder, 'video_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
        return metadata
    
    def _compile_video_from_frames(self, metadata):
        # Use the metadata to get the original video's fps, default to 30 if not available
        fps = metadata.get('fps', 30)
        
        # Create a writer for the output video
        writer = imageio.get_writer(self._output_name, fps=fps)
        
        # Sort and iterate over the saved frame images
        frame_files = sorted(os.listdir(self._frames_folder))  # Assumes only frame images are in this directory
        for frame_file in frame_files:
            if frame_file.endswith(".png"):  # Ensures we only process images
                frame_path = os.path.join(self._frames_folder, frame_file)
                frame = imageio.imread(frame_path)
                writer.append_data(frame)
        
        writer.close()
        print("Compiled frames into output video.")

    def _cleanup_temp_folder(self):
        if os.path.exists(self._frames_folder):
            shutil.rmtree(self._frames_folder)
            print("Temporary folder and its contents have been deleted.")

    def do_magic(self, prompt: str):
            self._prompt = prompt
            # Extract, transform, and save frames
            metadata = self._extract_and_transform_frames()
            
            # Compile the transformed frames into a video
            self._compile_video_from_frames(metadata)
            print("Done processing video.")

            self._cleanup_temp_folder()

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
    subprocess.run(command)


if __name__ == "__main__":
    input_video = "sample1.mpeg"
    output_video = input_video.split(".")[0] + "_SD" + ".mpeg"

    # Transform the video
    pipe = MildlyStableVideoPipeline(input_video, output_video)
    pipe.do_magic(prompt="aquarel painting")
    
    # At least on Windows, the output video does not necessarily work on default media player (works with ffplay, though)
    # Convert the video to mp4 to ensure compatibility
    convert_video_to_mp4(output_video, output_video.split(".")[0] + ".mp4")

