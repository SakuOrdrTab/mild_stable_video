# Stable diffusion for video blending a transformed frame one by one

The problem with stable diffusion and video is that it lacks context. There are multiple tools to introduce context to img2img, but none of them are currently feasible with local implementation. The quick key could be taking the last frame in a video sequence and just alter it a bit; this is a small demo what could be achieved by this.

For some examples, try this:
https://www.youtube.com/watch?v=G8UVR4xrd9g


## Installation

Just install the modules in requirements.txt, but as usual, the packages pertaining to tensor operations are highly platform dependant.

I would do like this:

Create a venv:

```
python -m venv .venv
```

Activate:

```
.\.venv\Scripts\activate
```

Install normal dependancies:

```
pip install -r requirements.txt
```

Install imageio, torch and etc:

```
pip install imageio imageio-ffmpeg
## install torch with https://pytorch.org/get-started/locally/

pip install accelerate
pip install diffusers
```
Takes a bit tweaking every time

## Using

Just place your video file in the appropriate place. Script uses ffmpeg for the last conversion of output to more usual mp4 so you should have that in your path, at least in windows. Currently the results are not very impressive, but I have not had much time tweaking with the params and all. Happy coding!
