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
    def __init__(self, video_name: str,
                output_name : str,
                guidance_scale : int = 7,
                image_strength : float = 0.1,
                inferring_steps : int = 10,
                initial_frame_guidance_scale : int =15,
                initial_image_strength : float = 0.5,
                last_frame_weight : float = 0.8):
        self._video_name = video_name
        self._output_name = output_name
        self._frames_folder = os.path.join(".", "temp")

        self._model_name = "stabilityai/stable-diffusion-2-base"
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_name, safety_checker=None)

        # Use cuda (graphics processor) if possible, If not, you probably have to use different arguments for  pipe init, too:
        # revision="fp16"
        # torch_dtype=torch.float16
        self._pipe = self._pipe.to("cuda")

        self._pipe.scheduler = DDIMScheduler.from_config(self._pipe.scheduler.config)
        self.scheduler_name = self._pipe.scheduler.config._class_name

        self._prompt = "" # Default prompt

        self._guidance = guidance_scale
        self._strength = image_strength
        self._passes = inferring_steps
        self._initial_frame_guidance_scale = initial_frame_guidance_scale
        self._initial_image_strength = initial_image_strength
        self._last_frame_weight = last_frame_weight

        self._last_transformed_image = None

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
                                            strength=self._initial_image_strength,
                                            guidance_scale=self._initial_frame_guidance_scale).images[0]

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
            frame_to_process = self._blend_images(last_image=last_image_resized, new_frame=resized_frame)
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

    def _blend_images(self, last_image, new_frame):
        '''Blends the last frame (tranformed) with the current frame'''
        return cv2.addWeighted(last_image, self._last_frame_weight, new_frame, 1-self._last_frame_weight, 0)
    
    def _extract_and_transform_frames(self):
        """Extracts frames from the video file and writes the files in a temp folder
        if the system would crash or similar. The frames are run through stable diffusion 
        and saved. The metadata is also saved in a JSON file.

        Returns:
            json: metadata of the video
        """        
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
        """Compiles a video from the transformed frames and saves it to the output file.

        Args:
            metadata (json): Metadata from the original video.
        """        
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
        """Deletes the temporary folder and its contents.
        """        
        if os.path.exists(self._frames_folder):
            shutil.rmtree(self._frames_folder)
            print("Temporary folder and its contents have been deleted.")

    def do_magic(self, prompt: str):
        """Runs the video through stable diffusion with the prompt and saves the video result.

        Args:
            prompt (str): The prompt for altering the video.
        """        
        start_time = time.time()
        self._prompt = prompt
        # Extract, transform, and save frames
        metadata = self._extract_and_transform_frames()
        
        # Compile the transformed frames into a video
        self._compile_video_from_frames(metadata)
        print("Done processing video.")

        print(f"The processing took: {(time.time() - start_time)/60:.2f} minutes.")
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
    pipe = MildlyStableVideoPipeline(input_video, output_video, guidance_scale=20, image_strength=0.1, inferring_steps=20, initial_frame_guidance_scale=20, initial_image_strength=0.3, last_frame_weight=0.9)
    pipe.do_magic(prompt="cartoon")
    
    # At least on Windows, the output video does not necessarily work on default media player (works with ffplay, though)
    # Convert the video to mp4 to ensure compatibility
    convert_video_to_mp4(output_video, output_video.split(".")[0] + ".mp4")

