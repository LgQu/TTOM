#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a single WanVideo given its pid.
"""

import os
import sys
import json
import argparse
import torch
from diffsynth import save_video
from ttt_lm.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import pandas as pd

# ----------------------------------------------------------------------
# 1. Build pipeline
# ----------------------------------------------------------------------
# model_base = "/home/ziyangw/temp/vgen/ttt-lm/hf_models/Wan-AI/Wan2.1-T2V-14B"
model_base = "/scratch/e1351271/video_gen/DiffSynth-Studio/models/Wan-AI/Wan2.1-VACE-14B"
import glob
dit_model = glob.glob(os.path.join(model_base, "*.safetensors"))

def build_pipeline() -> WanVideoPipeline:
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        skip_download=True,
        model_configs=[
            ModelConfig(path=dit_model),
            ModelConfig(
                path=os.path.join(model_base, "models_t5_umt5-xxl-enc-bf16.pth")
            ),
            ModelConfig(
                path=os.path.join(model_base, "Wan2.1_VAE.pth")
            ),
        ],
        tokenizer_config=ModelConfig(
            path="/scratch/e1351271/video_gen/DiffSynth-Studio/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
        )
    )
    # pipe.enable_vram_management()
    return pipe


# ----------------------------------------------------------------------
# 2. Generate a single video
# ----------------------------------------------------------------------
def generate_single_video(
    cache_path: str,
    out_dir: str,
    negative_prompt: str,
    seed: int = 42,
    pid: int = None,
    prompt: str = None,
    jsd_loss_weight: float = 0,
    com_loss_weight: float = 0,
    min_loss_value: float = 0.05,
    max_guidance_step: int = 5,
    max_lora_step: int = 30,
    max_iter: int = 10,
    target_modules: str = "q,k,v",
    guidance_type: str = "lora",
    skip_existed_prompt: bool = True,
    num_images: int = 0,
    load_lora_weight = False,
    update_lora_weight = False,
    save_lora_weight: bool = False,
    test_mode=False,
    save_mask=False,
):
    
    if pid is not None:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        entry = cache_data.get(str(pid))
        if entry is None:
            print(f"[ERROR] pid {pid} not found in JSON cache.")
            sys.exit(1)

        original_prompt = entry["original_prompt"]
        enriched_prompt = entry["enriched_prompt"]
        prompt = enriched_prompt

        objects = entry.get("objects", {})
        insts_prompts = [[(v.get("name") or v.get("object") or "")] for k, v in sorted(objects.items(), key=lambda kv: int(kv[0]))]
        tag = "[" + ",".join(p[0] for p in insts_prompts) + "]"
        layout = entry["layout"]["frames"]
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"pid{pid}_{tag}.mp4")
        if os.path.exists(out_path) and skip_existed_prompt:
            print(f"[SKIP] {out_path} already exists.")
            return
 
    else:
        print("[ERROR] Either pid or prompt must be provided.")
        sys.exit(1)

    pipe = build_pipeline()

    print(f"\n{'-'*40}\nGenerating video for pid={pid}")
    # Convert insts_prompts list to a consistent string format
    insts_str = str(insts_prompts).replace('"', "'")  # Ensure consistent single quotes
    attn_map_save_path = f"data/attn_maps/wan2.1-t2v-14b/pid{pid}_insts{insts_str}.pt"
    print(f"prompt: {prompt}")
    print(f"insts_prompts: {insts_prompts}")
    print(f"attn_map_save_path: {attn_map_save_path}")
    target_layers = tuple(range(40))
    video = pipe(
        prompt=prompt,
        pid=pid,
        guidance_type=guidance_type,
        negative_prompt=negative_prompt,
        insts_prompts=insts_prompts,
        layout=layout,
        target_layers=target_layers,
        seed=seed,
        tiled=False,
        jsd_loss_weight = jsd_loss_weight,
        com_loss_weight = com_loss_weight,
        min_loss_value=min_loss_value,
        max_iter=max_iter,
        max_guidance_step=max_guidance_step,
        target_modules=target_modules,
        max_lora_step=max_lora_step,
        save_attn_map=True,
        attn_map_save_path=attn_map_save_path,
        # tea_cache_l1_thresh=0.1,
        # tea_cache_model_id="Wan2.1-T2V-14B",
        save_lora_weight=save_lora_weight,
        load_lora_weight = load_lora_weight,
        update_lora_weight = update_lora_weight,
        test_mode=test_mode,
        save_mask=save_mask,
    )

    if video is None:
        print("Only lora weight is saved!")
        return

    if guidance_type == "using_existed_lora":
        if pid is not None:
            out_path = f"/scratch/e1351271/video_gen/lora_ckpts/pid{pid}.mp4"
    elif guidance_type == "none" and pid is None:
        out_path = f"/scratch/e1351271/video_gen/lora_ckpts/prompt[{prompt[:20]}]_origin.mp4"
        
    save_video(video, out_path, fps=16, quality=5)
    print(f"[DONE] pid {pid} saved to {out_path}")


# ----------------------------------------------------------------------
# 3. Command-line interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate a WanVideo for a given pid.")
    parser.add_argument("--pid", type=int, default=None, help="Sample ID to generate")
    parser.add_argument("--prompt", type=str, default=None, help="Sample ID to generate")
    parser.add_argument("--jsd_loss_weight", type=float, default=1.0, help="Sample ID to generate")
    parser.add_argument("--com_loss_weight", type=float, default=0.0, help="Sample ID to generate")
    parser.add_argument(
        "--cache_type",
        default="cache_train_motion_gpt-4o",
        help="Name of the cache folder",
    )
    parser.add_argument(
        "--guidance_type",
        default="none",
        help="guidance_type",
    )
    parser.add_argument(
        "--save_lora_weight",
        type=bool,
        default=False,
        help="save_lora_weight",
    )
    parser.add_argument(
        "--save_mask",
        type=bool,
        default=False,
        help="save_mask",
    )
    parser.add_argument('--skip_existed_prompt',
        type=bool,
        default=False,
        help="skip_existed_prompt",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o",
        help="target_modules",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=8,
        help="max_iter",
    )
    parser.add_argument(
        "--max_guidance_step",
        type=int,
        default=5,
        help="max_guidance_step",
    )
    parser.add_argument(
        "--max_lora_step",
        type=int,
        default=5,
        help="max_lora_step",
    )
    parser.add_argument(
        "--min_loss_value",
        type=float,
        default=0.03,
        help="min_loss_value",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=0,
        help="enable to activate t2i mode",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
    )
    parser.add_argument(
        "--strat_id",
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_json = f"{base}/cache/{args.cache_type}.json"
    out_dir = f"{base}/data/t2v_compbench/{args.cache_type}/{args.prefix}_wan_enriched_lora32_jsdGs_g{args.max_guidance_step}_ls{args.max_lora_step}_i{args.max_iter}_[{args.target_modules}]"
    
    if args.guidance_type == "none":
        out_dir = f"{base}/data/t2v_compbench/{args.cache_type}/wan_full_enriched_wo"

    negative_prompt = (
        "vivid tones, overexposed, static, blurry details, subtitles, artistic style, artwork, painting, "
        "still frame, grayish overall, worst quality, low quality, JPEG artifacts, ugly, deformed, "
        "extra fingers, poorly drawn hands, poorly drawn face, distorted, disfigured, malformed limbs, "
        "fused fingers, immobile image, cluttered background, three legs, crowded background, walking backward, "
        "monochrome, mutated hands and fingers, poorly rendered limbs, missing arms, missing legs, duplicate limbs, "
        "bad anatomy, unrealistic proportions, extra limbs, tiling, watermark, text, blurry, grainy, "
        "long neck, bad hands, mutated face, asymmetrical eyes, bad perspective, unnatural pose"
    )

    load_lora_weight = False
    update_lora_weight = False
    save_lora_weight = args.save_lora_weight
    test_mode = False
    save_mask = args.save_mask

    if args.strat_id == 1:
        load_lora_weight = True
        update_lora_weight = False
        save_lora_weight = False
        test_mode = True
    elif args.strat_id == 2:
        load_lora_weight = True
        update_lora_weight = True
        save_lora_weight = False
        test_mode = True
    elif args.strat_id == 0:
        load_lora_weight = False
        update_lora_weight = True
        save_lora_weight = False
        test_mode = True 

    print(f"[load | update | save]: [{load_lora_weight} | {update_lora_weight} | {save_lora_weight}]")
    print(f"save_mask: {save_mask}")

    print("--skip_existed_prompt", args.skip_existed_prompt)
    generate_single_video(
        cache_path=cache_json,
        pid=args.pid,
        prompt=args.prompt,
        out_dir=out_dir,
        negative_prompt=negative_prompt,
        seed=42,
        jsd_loss_weight = args.jsd_loss_weight,
        com_loss_weight = args.com_loss_weight,
        min_loss_value=args.min_loss_value,
        max_guidance_step=args.max_guidance_step,
        max_iter=args.max_iter,
        target_modules=args.target_modules,
        max_lora_step=args.max_lora_step,
        guidance_type=args.guidance_type,
        skip_existed_prompt=args.skip_existed_prompt,
        num_images=args.num_images,
        load_lora_weight=load_lora_weight,
        update_lora_weight=update_lora_weight,
        save_lora_weight=save_lora_weight,
        test_mode=test_mode,
        save_mask=save_mask,
    )


if __name__ == "__main__":
    main()
