"""
LVD-guided CogVideoX generation module.
Uses LVD_CogVideoXPipeline with layout-guided backward guidance for compositional video generation.
"""
import os
import sys
import json
import logging
import torch
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from dsl.model.cogvideox.controllable_pipeline_cogvideox import LVD_CogVideoXPipeline
from dsl.utils.cache import get_cache, cache_init, set_cache_path, get_objects_from_cache
from dsl.utils.layout_handler import interpolate_layout_boxes

logging.basicConfig(level=logging.INFO)

RESOLUTION_MAP = {
    "cogvideox-5b": (720, 480),
    "cogvideox-2b": (720, 480),
    "cogvideox1.5-5b": (1360, 768),
}


def generate_video(
    prompt: str = "",
    prompt_id: str = "0",
    model_path: str = "THUDM/CogVideoX-5b",
    generate_type: str = "t2v",
    cache_path: str = "cache/cache_4_motion_binding_gpt-4o.json",
    output_path: str = "data/out/video.mp4",
    num_frames: int = 49,
    fps: int = 8,
    num_inference_steps: int = 50,
    seed: int = 42,
    enable_cpu_offload: bool = True,
    guidance_scale: float = 6.0,
    guidance_type: str = "main",
    return_guidance_saved_attn: bool = False,
    guidance_start_step: int = 0,
    guidance_end_step: int = 10,
    backward_guidance_kwargs: Optional[dict] = None,
    energy_function_kwargs: Optional[dict] = None,
    **kwargs,
):
    if backward_guidance_kwargs is None:
        backward_guidance_kwargs = {}
    if energy_function_kwargs is None:
        energy_function_kwargs = {}

    set_cache_path(cache_path)
    cache_init()
    cache = get_cache()

    if prompt_id not in cache:
        raise ValueError(f"prompt_id '{prompt_id}' not found in cache: {cache_path}")

    entry = cache[prompt_id]
    prompt = entry.get("enriched_prompt", entry.get("original_prompt", prompt))
    objects = entry.get("objects", {})

    layout = entry.get("layout", {})
    frames_data = layout.get("frames", {})

    model_name = model_path.split("/")[-1].lower()
    width, height = RESOLUTION_MAP.get(model_name, (720, 480))

    latent_frames = (num_frames - 1) // 4 + 1
    bboxes = interpolate_layout_boxes(layout, num_interp_frames=latent_frames)
    backward_guidance_kwargs["bboxes"] = bboxes

    dtype = torch.bfloat16
    pipe = LVD_CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    if enable_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    generator = torch.Generator().manual_seed(seed)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    video_frames = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        objects=objects,
        prompt_id=prompt_id,
        guidance_type=guidance_type,
        guidance_start_step=guidance_start_step,
        guidance_end_step=guidance_end_step,
        backward_guidance_kwargs=backward_guidance_kwargs,
        energy_function_kwargs=energy_function_kwargs,
        return_guidance_saved_attn=return_guidance_saved_attn,
        seed=seed,
    ).frames[0]

    export_to_video(video_frames, output_path, fps=fps)
    logging.info(f"Video saved to {output_path}")
    return output_path
