"""
Visualize cross-attention maps from CogVideoX inference.

Loads saved attention tensors and generates heatmap overlays
showing per-object attention distributions across frames.

Usage:
    python utils/visualize_attn_maps.py \
        --pid 0 \
        --step_id 4 \
        --layer_id 20 \
        --inst_id 0 \
        --attn_dir data/attn_maps/cogvideox \
        --seed 42 \
        --guidance_type none \
        --save
"""
import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_attention_maps(attn_dir, step_id):
    path = os.path.join(attn_dir, f"attn_matrix_step_{step_id}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Attention map not found: {path}")
    return torch.load(path, map_location="cpu")


def visualize_single_layer(
    attn_maps,
    layer_id,
    inst_id,
    latent_h=30,
    latent_w=45,
    num_frames=13,
    save_path=None,
):
    if layer_id not in attn_maps:
        available = list(attn_maps.keys())
        raise KeyError(f"Layer {layer_id} not found. Available: {available}")

    attn = attn_maps[layer_id]
    if isinstance(attn, (list, tuple)):
        attn = attn[0]

    head, seq_len, tokens = attn.shape
    attn_avg = attn.mean(dim=0)  # [seq_len, tokens]

    attn_inst = attn_avg[:, inst_id]  # [seq_len]
    attn_inst = attn_inst.view(num_frames, latent_h, latent_w)

    ncols = min(num_frames, 7)
    nrows = (num_frames + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for t in range(num_frames):
        ax = axes[t]
        frame_attn = attn_inst[t].numpy()
        frame_attn = (frame_attn - frame_attn.min()) / (frame_attn.max() - frame_attn.min() + 1e-8)
        ax.imshow(frame_attn, cmap="jet", interpolation="bilinear")
        ax.set_title(f"Frame {t}", fontsize=9)
        ax.axis("off")

    for t in range(num_frames, len(axes)):
        axes[t].axis("off")

    plt.suptitle(f"Layer {layer_id}, Token {inst_id}", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize CogVideoX attention maps")
    parser.add_argument("--pid", type=str, required=True)
    parser.add_argument("--step_id", type=int, required=True)
    parser.add_argument("--layer_id", type=int, required=True)
    parser.add_argument("--inst_id", type=int, default=0, help="Token/instance index to visualize")
    parser.add_argument("--attn_dir", type=str, default="data/attn_maps/cogvideox")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_type", type=str, default="none")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    full_attn_dir = os.path.join(
        args.attn_dir,
        f"prompt{args.pid}_seed{args.seed}_guidance{args.guidance_type}"
    )

    attn_maps = load_attention_maps(full_attn_dir, args.step_id)

    save_path = None
    if args.save:
        save_path = os.path.join(
            full_attn_dir,
            f"vis_step{args.step_id}_layer{args.layer_id}_inst{args.inst_id}.png"
        )

    visualize_single_layer(
        attn_maps,
        layer_id=args.layer_id,
        inst_id=args.inst_id,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
