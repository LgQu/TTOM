"""
Attention map generation for CogVideoX.

Runs inference with attention map saving enabled, outputting per-step
cross-attention probability tensors for analysis and visualization.

Usage:
    python generation/get_attnmap.py \
        --pid 0 \
        --cache_type 4_motion_binding_gpt-4o \
        --model_path THUDM/CogVideoX-5b \
        --seed 42 \
        --guidance_type none \
        --output_dir data/attn_maps/cogvideox
"""
import sys
import os
import argparse
import json
import logging
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusers import CogVideoXDDIMScheduler
from diffusers.utils import export_to_video
from dsl.model.cogvideox.controllable_pipeline_cogvideox import LVD_CogVideoXPipeline
from dsl.utils.cache import get_cache, cache_init, set_cache_path
from dsl.utils.layout_handler import interpolate_layout_boxes

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate attention maps from CogVideoX")
    parser.add_argument("--pid", type=str, required=True)
    parser.add_argument("--cache_type", type=str, default="4_motion_binding_gpt-4o")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-5b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_type", type=str, default="none")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--output_dir", type=str, default="data/attn_maps/cogvideox")
    parser.add_argument("--guidance_attn_strategy", type=str, default="middle_2")
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--set_amf_loss", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    cache_path = f"cache/cache_{args.cache_type}.json"
    set_cache_path(cache_path)
    cache_init()
    cache = get_cache()

    entry = cache[args.pid]
    prompt = entry.get("enriched_prompt", entry["original_prompt"])
    objects = entry.get("objects", {})
    layout = entry.get("layout", {})

    latent_frames = (args.num_frames - 1) // 4 + 1
    bboxes = interpolate_layout_boxes(layout, num_interp_frames=latent_frames)

    dtype = torch.bfloat16
    pipe = LVD_CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    generator = torch.Generator().manual_seed(args.seed)

    attn_dir = os.path.join(
        args.output_dir,
        f"prompt{args.pid}_seed{args.seed}_guidance{args.guidance_type}"
    )
    os.makedirs(attn_dir, exist_ok=True)

    backward_guidance_kwargs = {
        "bboxes": bboxes,
        "loss_scale": 1.0,
        "loss_threshold": 12.0,
        "set_amf_loss": args.set_amf_loss,
        "max_iter": args.max_iter,
        "lr": args.lr,
        "verbose": True,
        "clear_cache": True,
        "guidance_attn_strategy": args.guidance_attn_strategy,
        "guidance_attn_keys": {},
        "set_latents_norm": False,
    }

    video_frames = pipe(
        prompt=prompt,
        height=480,
        width=720,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        objects=objects,
        prompt_id=args.pid,
        guidance_type=args.guidance_type,
        guidance_start_step=args.guidance_start_step,
        guidance_end_step=args.guidance_end_step,
        backward_guidance_kwargs=backward_guidance_kwargs if args.guidance_type == "main" else None,
        energy_function_kwargs={},
        return_guidance_saved_attn=True,
        seed=args.seed,
    ).frames[0]

    video_path = os.path.join(attn_dir, f"video_{args.pid}.mp4")
    export_to_video(video_frames, video_path, fps=8)

    print(f"\nAttention maps saved to: {attn_dir}")
    print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    main()
