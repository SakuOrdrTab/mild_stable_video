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
    def __init__(self, ):
        self._model_name = "stabilityai/stable-diffusion-2-base"
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_name, safety_checker=None)

        # Use cuda (graphics processor) if possible, If not, you probably have to use different arguments for  pipe init, too:
        # revision="fp16"
        # torch_dtype=torch.float16
        self._pipe = self._pipe.to("cuda")

        self._pipe.scheduler = DDIMScheduler.from_config(self._pipe.scheduler.config)
        self.scheduler_name = self._pipe.scheduler.config._class_name

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
            frame_to_process = self._blend_images(last_image=self._last_transformed_image, new_frame=resized_frame)
        else:
            # return a more processed image
            frame_to_process = resized_frame
            return self._first_frame(frame_to_process)
        
        pil_image = Image.fromarray(frame_to_process)

        # Process the PIL image with the pipeline
        with torch.no_grad():  # Use torch.no_grad() to reduce memory usage
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

    def _blend_images(self, last_image, new_frame):
        '''Blends the last frame (tranformed) with the current frame'''
        return cv2.addWeighted(last_image, self._last_frame_weight, new_frame, 1-self._last_frame_weight, 0)
    
    def _open_video(self, video_path):
        """Gets info of the video file, so it can be saved in the same format.

        Args:
            video_path (str): path to the video file.

        Returns:
            reader, metadata, video_format: image.io reader, metadata and video format
        """        
        try:
            reader = imageio.get_reader(video_path)
            metadata = reader.get_meta_data()
            video_format = metadata.get('format', None)  # Get format or None if missing
            return reader, metadata, video_format
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error reading video: {e}")
            return None, None
    
    def _extract_and_transform_frames(self):
        """Extracts frames from the video file and writes the files in a temp folder
        if the system would crash or similar. The frames are run through stable diffusion 
        and saved. The metadata is also saved in a JSON file.

        Returns:
            json: metadata of the video
        """        
        reader, metadata, self._video_format = self._open_video(self._video_name)
        # If no metadata, then there is no video to process
        if not metadata:
            return
        
        # Get count of frames, if it is available
        frame_count = reader.count_frames()
        print(f"There are {frame_count} frames to process in the video.")
        
        # Ensure frames folder exists
        if not os.path.exists(self._frames_folder):
            os.makedirs(self._frames_folder)

        # Iterate over, transform, and save each frame as an image
        for i, frame in enumerate(reader):
            image_path = os.path.join(self._frames_folder, f"frame_{i:04d}_transformed.png")
            if not os.path.exists(image_path):  # Only process and save if the frame hasn't been saved before
                transformed_frame = self._transform_frame(frame)
                imageio.imwrite(image_path, transformed_frame)
                print(f"Processed and saved frame {i}/{frame_count}.")
            else:
                print(f"Frame {i} already processed and saved.")

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
        
        try:
            writer = imageio.get_writer(self._output_name, format=self._video_format)
        except Exception as e1:
            print(f"Warning: Original format {self._video_format} not supported by imageio. Using default.")
            try: 
                writer = imageio.get_writer(self._output_name, fps=fps)
            except Exception as e2:
                print(f"Error creating video writer: {e2}")
                return
        
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
            try:
                shutil.rmtree(self._frames_folder)
            except Exception as e:
                print(f"Error deleting temp folder: {e}")
            print("Temporary folder and its contents have been deleted.")

    def do_magic(self,
                video_name: str,
                output_name : str,
                prompt: str = "",
                negative_prompt: str = "",
                guidance_scale : int = 7,
                image_strength : float = 0.1,
                inferring_steps : int = 10,
                initial_frame_guidance_scale : int =15,
                initial_image_strength : float = 0.5,
                last_frame_weight : float = 0.6):
        """Runs the video through stable diffusion with the prompt and saves the video result.

        Args:
            prompt (str): The prompt for altering the video.
        """        
        start_time = time.time()
        self._video_name = video_name
        self._output_name = output_name
        self._guidance = guidance_scale
        self._strength = image_strength
        self._passes = inferring_steps
        self._frames_folder = os.path.join(".", "temp")
        self._initial_frame_guidance_scale = initial_frame_guidance_scale
        self._initial_image_strength = initial_image_strength
        self._last_frame_weight = last_frame_weight
        self._prompt = prompt
        self._negative_prompt = negative_prompt

        # Extract, transform, and save frames
        metadata = self._extract_and_transform_frames()
        
        # Compile the transformed frames into a video
        self._compile_video_from_frames(metadata)
        print("Done processing video.")

        print(f"The processing took: {(time.time() - start_time)/60:.2f} minutes.")
        self._cleanup_temp_folder()

if __name__ == "__main__":
    test_pipe = MildlyStableVideoPipeline()
    # Force some attributes to parent class to test function
    test_pipe._prompt = "japanese wood painting"
    test_pipe._negative_prompt = ""
    test_pipe._initial_frame_guidance_scale = 15
    test_pipe._initial_image_strength = 0.7

    # Load sample initial image
    init_image = np.asarray(Image.open("sample_image.jpg"))
    # Show result that the pipeline returns as an array
    Image.fromarray(test_pipe._first_frame(init_image)).show()
