# flowWarpedVideoPipeline.py
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler
from PIL import Image

# By just importing a different base image tranformation pipeline, we can swap out the underlying model.

# from mildlyStableVideoPipeline import MildlyStableVideoPipeline
# SD3 does not work, as it uses a tranformer backbnone instead of unet
from mildlyStableXLVideoPipeline import MildlyStableXLVideoPipeline as MildlyStableVideoPipeline

def _encode_latents(vae, pil_image, device, dtype):
    """Encode RGB PIL image -> latent z0 (scaled) deterministically."""
    t = torch.from_numpy(np.array(pil_image)).to(device=device, dtype=dtype) / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)          # (1,3,H,W)
    t = (t - 0.5) * 2                            # [-1, 1]
    with torch.no_grad():
        posterior = vae.encode(t)
        z = posterior.latent_dist.mode()         # deterministic (use .mean() if preferred)
    return z * vae.config.scaling_factor         # (1,4,h,w)


def _warp_tensor_with_flow(x, flow_hw, device, dtype, mode="bilinear"):
    """
    x: (B,C,h,w) tensor (noise or latents) on device
    flow_hw: (H,W,2) float32, pixel flow (u,v) at image resolution.
    Downscales to latent res and builds a sampling grid for grid_sample.
    """
    B, C, h, w = x.shape
    H, W, _ = flow_hw.shape

    # resize flow to latent size
    flow_small = cv2.resize(flow_hw, (w, h), interpolation=cv2.INTER_LINEAR)

    # convert pixel flow -> normalized grid offsets in [-1, 1]
    fx = flow_small[..., 0] / ((w - 1) / 2.0)
    fy = flow_small[..., 1] / ((h - 1) / 2.0)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack(
        (
            xx - torch.from_numpy(fx).to(device=device, dtype=dtype),
            yy - torch.from_numpy(fy).to(device=device, dtype=dtype),
        ),
        dim=-1,
    )  # (h,w,2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    return F.grid_sample(x, grid, mode=mode, padding_mode="border", align_corners=True)


class FlowWarpedVideoPipeline(MildlyStableVideoPipeline):
    """
    Same external API as your MildlyStableVideoPipeline, but enforces temporal coherence via:
      - optical-flow-warped initial noise
      - (optional) light latent-content blending across frames
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name += " + Flow Warping"

        # Swap to a robust scheduler for img2img
        self._pipe.scheduler = EulerDiscreteScheduler.from_config(self._pipe.scheduler.config)

        # caches for temporal coherence
        self._prev_noise = None      # torch tensor (1,4,h,w)
        self._prev_frame = None      # np.uint8 (H,W,3) RGB
        self._prev_z0 = None         # torch tensor (1,4,h,w)

        # Optical flow: DIS if available (opencv-contrib), else Farnebäck
        self._use_dis = False
        try:
            # OpenCV sometimes exposes DIS at cv2.DISOpticalFlow_create, sometimes in cv2.optflow
            if hasattr(cv2, "DISOpticalFlow_create"):
                self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                self._use_dis = True
            elif hasattr(cv2, "optflow") and hasattr(cv2.optflow, "createOptFlow_DIS"):
                self._dis = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
                self._use_dis = True
            else:
                self._dis = None
        except Exception:
            self._dis = None
            self._use_dis = False

        # strength for latent-content blending across frames (flow-warped)
        # reuse your last_frame_weight but soften it (latent blend should be gentle)
        self._beta_latent_blend = 0.25 * getattr(self, "_last_frame_weight", 0.6)

    def _calc_flow(self, prev_rgb_uint8, curr_rgb_uint8):
        prev_gray = cv2.cvtColor(prev_rgb_uint8, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_rgb_uint8, cv2.COLOR_RGB2GRAY)
        if self._use_dis and self._dis is not None:
            flow = self._dis.calc(prev_gray, curr_gray, None)   # (H,W,2) float32
        else:
            # Farnebäck fallback (slower, but fine)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=21,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        return flow

    def _prep_timesteps(self, num_inference_steps, strength):
        """
        Mirror Diffusers' img2img timestep logic so our custom latents align:
          init_t = int(num_steps * strength)
          t_start = num_steps - init_t
          first_timestep = timesteps[t_start]
        """
        self._pipe.scheduler.set_timesteps(num_inference_steps, device=self._pipe.device)
        timesteps = self._pipe.scheduler.timesteps
        init_t = int(num_inference_steps * float(strength))
        t_start = max(num_inference_steps - init_t, 0)
        return timesteps, t_start

    @torch.no_grad()
    def _stylize_with_flow(self, frame_rgb_uint8, prompt, negative_prompt, strength, guidance, steps, seed):
        """
        Core per-frame routine: encode -> build/warp noise -> add noise at t0 -> pipeline denoise (with latents).
        """
        device = self._pipe.device
        dtype = next(self._pipe.unet.parameters()).dtype

        # Ensure dimensions are multiples of 8 (SD constraint). Your 640x480 is fine.
        H, W, _ = frame_rgb_uint8.shape
        if (H % 8) != 0 or (W % 8) != 0:
            H8, W8 = (H // 8) * 8, (W // 8) * 8
            frame_rgb_uint8 = cv2.resize(frame_rgb_uint8, (W8, H8), interpolation=cv2.INTER_AREA)
            H, W = H8, W8

        pil = Image.fromarray(frame_rgb_uint8)

        # 1) Encode init image -> z0 (deterministic)
        z0 = _encode_latents(self._pipe.vae, pil, device, dtype)  # (1,4,h,w)
        h, w = z0.shape[-2:]

        # 2) Build or warp starting noise (shape matches z0)
        gen = torch.Generator(device=device).manual_seed(int(seed))
        if self._prev_noise is None:
            noise = torch.randn((1, 4, h, w), generator=gen, device=device, dtype=dtype)
        else:
            flow = self._calc_flow(self._prev_frame, frame_rgb_uint8)  # (H,W,2)
            noise = _warp_tensor_with_flow(self._prev_noise, flow, device, dtype, mode="bilinear")

        # 3) Optional: light latent-content blend (flow-warped previous z0)
        beta = float(self._beta_latent_blend)
        if self._prev_z0 is not None and beta > 0.0:
            flow = self._calc_flow(self._prev_frame, frame_rgb_uint8)
            prev_z0_warp = _warp_tensor_with_flow(self._prev_z0, flow, device, dtype, mode="bilinear")
            z0 = (1.0 - beta) * z0 + beta * prev_z0_warp

        # 4) Compute starting timestep and create initial *noisy* latents
        timesteps, t_start = self._prep_timesteps(steps, strength)
        start_t = timesteps[t_start]  # scheduler step to which we add noise
        latents = self._pipe.scheduler.add_noise(z0, noise, start_t.unsqueeze(0))
        # or: latents = self._pipe.scheduler.add_noise(z0, noise, [start_t])


        # 5) Run the *pipeline* with our custom latents (it will skip encoding & noise add)
        image = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil,  # still pass for validation; latents take precedence
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
            latents=latents,
        ).images[0]

        # cache for next frame
        self._prev_noise = noise.detach()
        self._prev_frame = frame_rgb_uint8.copy()
        self._prev_z0 = z0.detach()

        return np.array(image)

    def _transform_frame(self, frame):
        """
        Overrides parent: uses flow-warped noise + latent blending.
        Honors your first-frame special params by switching strength/guidance on first call.
        """
        # Keep your 640x480 resize for consistency with the rest of the project
        resized = cv2.resize(frame, (640, 480))

        # pick params: first frame gets your stronger prompt guidance & strength
        is_first = self._prev_frame is None
        steps = int(self._passes)
        if is_first:
            strength = float(getattr(self, "_initial_image_strength", self._strength))
            guidance = float(getattr(self, "_initial_frame_guidance_scale", self._guidance))
        else:
            strength = float(self._strength)
            guidance = float(self._guidance)

        # Fixed seed per run (you set torch.manual_seed elsewhere too). For reproducibility across frames,
        # we seed the *initial* noise; temporal correlation comes from warping.
        seed = 66

        transformed = self._stylize_with_flow(
            frame_rgb_uint8=resized,
            prompt=self._prompt,
            negative_prompt=self._negative_prompt,
            strength=strength,
            guidance=guidance,
            steps=steps,
            seed=seed,
        )

        self._last_transformed_image = transformed.copy()
        return transformed


if __name__ == "__main__":
    # quick smoke test on a single image
    import numpy as np
    from PIL import Image

    test_pipe = FlowWarpedVideoPipeline()
    test_pipe._prompt = "japanese wood painting"
    test_pipe._negative_prompt = ""
    test_pipe._initial_frame_guidance_scale = 15
    test_pipe._initial_image_strength = 0.7
    test_pipe._guidance = 7.5
    test_pipe._strength = 0.35
    test_pipe._passes = 30

    init_image = np.asarray(Image.open("sample_image.jpg").convert("RGB"))
    Image.fromarray(test_pipe._transform_frame(init_image)).show()
