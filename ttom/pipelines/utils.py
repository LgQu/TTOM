import matplotlib.pyplot as plt
import numpy as np
import torch

def box_to_mask(box, canvas_w, canvas_h):
    """
    Convert box in [x0, y0, w, h] format to normalized mask (binary image)
    mask shape is (canvas_h, canvas_w), values are 0 or 1
    """
    x0, y0, w, h = box
    mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    x1 = int(round(x0))
    y1 = int(round(y0))
    x2 = int(round(x0 + w))
    y2 = int(round(y0 + h))

    x1 = max(0, min(x1, canvas_w))
    x2 = max(0, min(x2, canvas_w))
    y1 = max(0, min(y1, canvas_h))
    y2 = max(0, min(y2, canvas_h))

    mask[y1:y2, x1:x2] = 1.0
    return mask

def interpolate_instance_masks(instance_boxes, canvas_h, canvas_w, num_frames=21):
    keyframe_map = [0, 4, 8, 12, 16, 20]

    def box_to_mask(box, h, w):
        x0, y0, bw, bh = box
        mask = torch.zeros((h, w), dtype=torch.float32)
        x1, y1 = int(round(x0)), int(round(y0))
        x2, y2 = int(round(x0 + bw)), int(round(y0 + bh))
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        mask[y1:y2, x1:x2] = 1.0
        return mask

    result = {}

    for inst_id in sorted(instance_boxes.keys()):
        key_idxs = sorted(instance_boxes[inst_id].keys())
        full_mask = torch.zeros((num_frames, canvas_h, canvas_w), dtype=torch.float32)

        for i in range(len(key_idxs) - 1):
            kf0, kf1 = key_idxs[i], key_idxs[i + 1]
            if kf0 >= len(keyframe_map) or kf1 >= len(keyframe_map):
                continue

            idx0, idx1 = keyframe_map[kf0], keyframe_map[kf1]
            box0 = instance_boxes[inst_id][kf0]
            box1 = instance_boxes[inst_id][kf1]

            for t in range(idx0, idx1 + 1):
                alpha = (t - idx0) / (idx1 - idx0) if idx1 > idx0 else 0.0
                box_t = [
                    (1 - alpha) * box0[0] + alpha * box1[0],
                    (1 - alpha) * box0[1] + alpha * box1[1],
                    (1 - alpha) * box0[2] + alpha * box1[2],
                    (1 - alpha) * box0[3] + alpha * box1[3],
                ]
                full_mask[t] = box_to_mask(box_t, canvas_h, canvas_w)

        result[inst_id] = full_mask

    return result


def vis_layout(layout_list, output_path="layout_mask_vis.mp4"):
    import imageio
    import torch
    import numpy as np
    from matplotlib import cm
    from PIL import Image

    output_path = output_path
    fps = 5

    num_instances = len(layout_list)
    T = len(layout_list[0]["masks"])  # 21 frames
    H, W = layout_list[0]["masks"][0].shape  # Automatically get real dimensions, e.g. 480×720

    id_mask = torch.zeros((T, H, W), dtype=torch.int32)

    for idx, instance in enumerate(layout_list):
        masks = instance["masks"]  # List[Tensor(H, W)]
        for t in range(T):
            id_mask[t] += (masks[t] > 0).int() * (idx + 1)

    id_mask_np = id_mask.cpu().numpy()  # (T, H, W)

    cmap = cm.get_cmap("nipy_spectral", num_instances + 1)  # 0 is background
    frames = []

    for t in range(T):
        frame_id = id_mask_np[t]  # (H, W)
        rgba_img = cmap(frame_id / (num_instances + 1))[:, :, :3]  # normalize to [0,1]
        rgb_img = (rgba_img * 255).astype(np.uint8)  # [H, W, 3]

        # Use original dimensions directly, no resize
        frames.append(rgb_img)

    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")
    print(f"Video saved to {output_path} with size {W}x{H}")

def save_attention_heatmap_frames(
    attn_tensor: torch.Tensor,
    save_path: str,
    tag: str,
    frame_shape=(30, 52),
):
    """
    Save attention heatmap frames in (num_frames, H, W) format as images frame by frame.

    Args:
    - attn_tensor: Tensor of shape [num_frames, H, W]
    - save_path: Path to save heatmap
    - tag: File name prefix
    - frame_shape: (H, W) of each frame, can be used for validation
    """
    attn_tensor = attn_tensor.detach().to(torch.float32).cpu()
    num_frames, H, W = attn_tensor.shape

    assert (H, W) == frame_shape, f"Shape mismatch: expected {(H, W)}, got {(attn_tensor.shape[1], attn_tensor.shape[2])}"

    attn_tensor = attn_tensor.numpy()

    # Normalize each frame independently or jointly
    attn_min = attn_tensor.min()
    attn_max = attn_tensor.max()
    attn_tensor = (attn_tensor - attn_min) / (attn_max - attn_min + 1e-8)

    # Create subplots (horizontal layout)
    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 3, 3))
    if num_frames == 1:
        axes = [axes]  # Ensure it's iterable

    for i in range(num_frames):
        ax = axes[i]
        im = ax.imshow(attn_tensor[i], cmap="viridis", aspect="auto")
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    full_path = f"{save_path}_{tag}.png"
    plt.savefig(full_path, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    print(f"✅ Saved grid heatmap to {full_path}")


import os
import torch

def save_attention_maps_from_history(
    attn_map_history,
    save_path,
    progress_id,
    T=21, H_p=30, W_p=52,
    store_token_idx=False,
    batch_index=0,      # If B>1, select which sample
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Load/initialize and be compatible with old key name 'progs'
    if os.path.isfile(save_path):
        blob = torch.load(save_path, map_location="cpu")
        if "progs" in blob and "timestep" not in blob:
            blob["timestep"] = blob.pop("progs")
    else:
        blob = {"meta": {"T": T, "H_p": H_p, "W_p": W_p}, "timestep": {}}

    meta = blob.setdefault("meta", {})
    assert meta.get("T", T) == T and meta.get("H_p", H_p) == H_p and meta.get("W_p", W_p) == W_p, \
        f"Meta mismatch: existing={meta}, new=(T={T}, H_p={H_p}, W_p={W_p})"

    ts_slot = blob.setdefault("timestep", {}).setdefault(int(progress_id), {})
    layers_slot = ts_slot.setdefault("layers", {})

    def _pick_tensor(attn_maps):
        """Extract Tensor from Tensor / List[Tensor] / Dict"""
        if attn_maps is None:
            return None
        if torch.is_tensor(attn_maps):
            return attn_maps
        if isinstance(attn_maps, (list, tuple)):
            for x in reversed(attn_maps):            # Take the last one
                if torch.is_tensor(x):
                    return x
                if isinstance(x, dict):
                    for k in ("attn", "attention", "map", "tensor"):
                        v = x.get(k)
                        if torch.is_tensor(v):
                            return v
        if isinstance(attn_maps, dict):
            for k in ("attn", "attention", "map", "tensor"):
                v = attn_maps.get(k)
                if torch.is_tensor(v):
                    return v
        return None

    for layer_id in sorted(attn_map_history.keys()):
        attn_maps = attn_map_history[layer_id]
        attn_any = _pick_tensor(attn_maps)
        if attn_any is None:
            continue

        # Unify to (N,S,L)
        if attn_any.dim() == 4:
            B, N, S, L = attn_any.shape
            if batch_index >= B:
                batch_index = 0
            attn_nsl = attn_any[batch_index]
        elif attn_any.dim() == 3:
            N, S, L = attn_any.shape
            attn_nsl = attn_any
        else:
            continue

        if S != T * H_p * W_p:
            continue

        layer_slot = layers_slot.setdefault(int(layer_id), {})
        insts_slot = layer_slot.setdefault("insts", {})

        for n in range(N):
            attn_per_inst = attn_nsl[n]                # (S,L)
            token_mask = (attn_per_inst.sum(dim=0) > 0)
            if token_mask.any():
                attn_valid = attn_per_inst[:, token_mask]
                attn_avg = attn_valid.mean(dim=1)      # (S,)
                token_idx = torch.nonzero(token_mask, as_tuple=False).view(-1)
            else:
                attn_avg = attn_per_inst.mean(dim=1)   # (S,)
                token_idx = torch.arange(attn_per_inst.shape[1]) if store_token_idx else None

            attn_grid = attn_avg.view(T, H_p, W_p).to(dtype=torch.float32, device='cpu').contiguous()
            
            # Normalization processing
            attn_min = attn_grid.min()
            attn_max = attn_grid.max()
            if attn_max > attn_min:
                attn_grid = (attn_grid - attn_min) / (attn_max - attn_min)

            slot = {"attn_map": attn_grid}
            if store_token_idx and token_idx is not None:
                slot["token_idx"] = token_idx.to(torch.long, "cpu")
            insts_slot[int(n)] = slot

    torch.save(blob, save_path)
