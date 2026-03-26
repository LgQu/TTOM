"""
Evaluate mIoU between attention maps and object segmentation masks.

Computes mean Intersection-over-Union between cross-attention heatmaps
and ground-truth object masks (from layout boxes or detection results).

Usage:
    python utils/evaluate_miou.py \
        --attn_dir data/attn_maps/cogvideox \
        --cache_type 4_motion_binding_gpt-4o \
        --output_dir data/miou_summary \
        --pid 0 \
        --seed 42
"""
import sys
import os
import argparse
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dsl.utils.cache import get_cache, cache_init, set_cache_path
from dsl.utils.layout_handler import interpolate_layout_boxes
from dsl.utils.utils import scale_proportion


def compute_miou_for_prompt(
    attn_maps,
    bboxes,
    object_positions_id_dict,
    latent_h=30,
    latent_w=45,
    num_frames=13,
    threshold=0.5,
    upsample_scale=4,
):
    """Compute mIoU between attention maps and layout box masks."""
    H = latent_h * upsample_scale
    W = latent_w * upsample_scale

    results = {}
    layer_keys = sorted(attn_maps.keys())

    for layer_key in layer_keys:
        attn = attn_maps[layer_key]
        if isinstance(attn, (list, tuple)):
            attn = attn[0]

        head, seq_len, tokens_num = attn.shape
        attn_avg = attn.mean(dim=0)  # [seq_len, tokens]

        layer_ious = []
        for obj_idx, obj_boxes in enumerate(bboxes):
            obj_key = str(obj_idx)
            if obj_key not in object_positions_id_dict:
                continue

            token_ids = object_positions_id_dict[obj_key]
            obj_attn = attn_avg[:, token_ids].mean(dim=-1)
            obj_attn = obj_attn.view(num_frames, latent_h, latent_w)

            if upsample_scale != 1:
                obj_attn = torch.nn.functional.interpolate(
                    obj_attn.unsqueeze(0).unsqueeze(0).float(),
                    size=(num_frames, H, W),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze()

            frame_ious = []
            for t in range(min(num_frames, len(obj_boxes))):
                box = obj_boxes[t]
                if box is None:
                    continue

                mask = torch.zeros(H, W)
                x1, y1, x2, y2 = scale_proportion(box, H, W)
                mask[y1:y2, x1:x2] = 1.0

                attn_frame = obj_attn[t]
                attn_norm = (attn_frame - attn_frame.min()) / (attn_frame.max() - attn_frame.min() + 1e-8)
                attn_bin = (attn_norm > threshold).float()

                intersection = (attn_bin * mask).sum()
                union = ((attn_bin + mask) > 0).float().sum()
                iou = (intersection / (union + 1e-8)).item()
                frame_ious.append(iou)

            if frame_ious:
                layer_ious.append(np.mean(frame_ious))

        if layer_ious:
            results[layer_key] = np.mean(layer_ious)

    overall_miou = np.mean(list(results.values())) if results else 0.0
    return overall_miou, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate mIoU between attention maps and layout masks")
    parser.add_argument("--attn_dir", type=str, default="data/attn_maps/cogvideox")
    parser.add_argument("--cache_type", type=str, default="4_motion_binding_gpt-4o")
    parser.add_argument("--output_dir", type=str, default="data/miou_summary")
    parser.add_argument("--pid", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_type", type=str, default="none")
    parser.add_argument("--step_ids", type=str, default="0,4,8,12",
                        help="Comma-separated step IDs to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cache_path = f"cache/cache_{args.cache_type}.json"
    set_cache_path(cache_path)
    cache_init()
    cache = get_cache()

    entry = cache[args.pid]
    layout = entry.get("layout", {})
    num_frames = 49
    latent_frames = (num_frames - 1) // 4 + 1
    bboxes = interpolate_layout_boxes(layout, num_interp_frames=latent_frames)

    full_attn_dir = os.path.join(
        args.attn_dir,
        f"prompt{args.pid}_seed{args.seed}_guidance{args.guidance_type}"
    )

    step_ids = [int(s) for s in args.step_ids.split(",")]
    all_results = {}

    for step_id in step_ids:
        attn_path = os.path.join(full_attn_dir, f"attn_matrix_step_{step_id}.pt")
        if not os.path.exists(attn_path):
            print(f"[Skip] Step {step_id}: file not found")
            continue

        attn_maps = torch.load(attn_path, map_location="cpu")

        objects = entry.get("objects", {})
        obj_pos_dict = {}
        for obj_idx in objects:
            positions = objects[obj_idx].get("object_positions", [])
            token_ids = [t["id"] for pos in positions for t in pos.get("matched_tokens", [])]
            if token_ids:
                obj_pos_dict[obj_idx] = token_ids

        miou, per_layer = compute_miou_for_prompt(
            attn_maps, bboxes, obj_pos_dict,
            num_frames=latent_frames,
            threshold=args.threshold,
        )
        all_results[step_id] = {"miou": miou, "per_layer": {str(k): v for k, v in per_layer.items()}}
        print(f"Step {step_id}: mIoU = {miou:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"miou_pid{args.pid}_seed{args.seed}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
