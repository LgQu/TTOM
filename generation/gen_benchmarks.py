"""
Phase 2: Video generation with CogVideoX + layout-guided backward guidance (TTOM).

Generates videos conditioned on extracted metadata and layouts, with iterative
test-time optimization of cross-attention maps via latent guidance.

Usage:
    python generation/gen_benchmarks.py \
        --pid 0 \
        --cache_type 4_motion_binding_gpt-4o \
        --model_path THUDM/CogVideoX-5b \
        --guidance_type main \
        --guidance_attn_strategy middle_2 \
        --guidance_start_step 0 \
        --guidance_end_step 10 \
        --max_iter 5 \
        --lr 5e-3 \
        --set_amf_loss \
        --seed 42
"""
import sys
import os
import time
import glob
import argparse
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dsl.generation.lvd_cogvideo import generate_video
from dsl.utils.cache import set_cache_path

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Video generation with CogVideoX + TTOM guidance")

    parser.add_argument("--pid", type=str, required=True, help="Prompt ID to generate")
    parser.add_argument("--cache_type", type=str, default="4_motion_binding_gpt-4o",
                        help="Cache file name (without 'cache_' prefix and '.json' suffix)")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-5b")
    parser.add_argument("--generate_type", type=str, default="t2v")

    parser.add_argument("--guidance_type", type=str, default="main",
                        choices=["main", "none"], help="Guidance type: main or none")
    parser.add_argument("--guidance_attn_strategy", type=str, default="middle_2",
                        help="Attention layer selection strategy")
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=10)

    parser.add_argument("--max_iter", type=int, default=5, help="Max guidance iterations per step")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for latent optimization")
    parser.add_argument("--loss_scale", type=float, default=1.0)
    parser.add_argument("--loss_threshold", type=float, default=12.0)
    parser.add_argument("--set_amf_loss", action="store_true", help="Use AMF (Attention Motion Flow) loss")
    parser.add_argument("--set_latents_norm", action="store_true")

    parser.add_argument("--use_ciou_based_loss", action="store_true")
    parser.add_argument("--use_ratio_based_loss", action="store_true")
    parser.add_argument("--com_loss_scale", type=float, default=0.0)
    parser.add_argument("--region_loss_scale", type=float, default=0.0)
    parser.add_argument("--info_entropy_loss_scale", type=float, default=0.0)

    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_cpu_offload", action="store_true", default=True)

    parser.add_argument("--skip_existed_prompt", action="store_true", help="Skip if output already exists")
    parser.add_argument("--prefix", type=str, default="", help="Output directory prefix")
    parser.add_argument("--save_attn", action="store_true", help="Save attention maps")

    return parser.parse_args()


def main():
    args = parse_args()

    cache_path = f"cache/cache_{args.cache_type}.json"
    set_cache_path(cache_path)

    model_name = args.model_path.split("/")[-1]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    tag = f"g{args.guidance_type}_attn-{args.guidance_attn_strategy}_iter{args.max_iter}_lr{args.lr}"
    if args.set_amf_loss:
        tag += "_amf"
    if args.prefix:
        tag = f"{args.prefix}_{tag}"

    output_dir = os.path.join(
        "data", "benchmarks", args.cache_type,
        f"{model_name}_s{args.seed}", tag
    )
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{args.pid}_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    if args.skip_existed_prompt:
        existing = glob.glob(os.path.join(output_dir, f"{args.pid}_*.mp4"))
        if existing:
            print(f"[Skip] Output already exists for pid={args.pid}: {existing[0]}")
            return

    backward_guidance_kwargs = {
        "loss_scale": args.loss_scale,
        "loss_threshold": args.loss_threshold,
        "set_amf_loss": args.set_amf_loss,
        "max_iter": args.max_iter,
        "lr": args.lr,
        "verbose": True,
        "clear_cache": True,
        "guidance_attn_strategy": args.guidance_attn_strategy,
        "guidance_attn_keys": {},
        "set_latents_norm": args.set_latents_norm,
    }

    energy_function_kwargs = {
        "use_ciou_based_loss": args.use_ciou_based_loss,
        "use_ratio_based_loss": args.use_ratio_based_loss,
        "com_loss_scale": args.com_loss_scale,
        "region_loss_scale": args.region_loss_scale,
        "info_entropy_loss_scale": args.info_entropy_loss_scale,
    }

    print(f"\n{'='*60}")
    print(f"Generating video for pid={args.pid}")
    print(f"  Model: {args.model_path}")
    print(f"  Cache: {cache_path}")
    print(f"  Guidance: {args.guidance_type}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    generate_video(
        prompt_id=args.pid,
        model_path=args.model_path,
        generate_type=args.generate_type,
        cache_path=cache_path,
        output_path=output_path,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        enable_cpu_offload=args.enable_cpu_offload,
        guidance_scale=args.guidance_scale,
        guidance_type=args.guidance_type,
        return_guidance_saved_attn=args.save_attn,
        guidance_start_step=args.guidance_start_step,
        guidance_end_step=args.guidance_end_step,
        backward_guidance_kwargs=backward_guidance_kwargs,
        energy_function_kwargs=energy_function_kwargs,
    )

    print(f"\nVideo saved to: {output_path}")


if __name__ == "__main__":
    main()
