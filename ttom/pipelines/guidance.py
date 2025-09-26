
import os
from typing import Dict, List
import torch
import numpy as np
import torch.nn.functional as F
from ttom.pipelines import utils
import gc

def save_attention_maps_from_history(
    attn_map_history,
    save_path,
    progress_id,          # This parameter name is still used here; will be used as the timestep key
    T=21, H_p=30, W_p=52,
    store_token_idx=False,
):
    """
    Save attention maps as .pt/.pth files with structure:
    {
      'meta': {'T':..., 'H_p':..., 'W_p':...},
      'timestep': {
        <timestep_id>: {
          'layers': {
            <layer_id>: {
              'insts': {
                <inst_id>: {
                  'attn_map': FloatTensor[T, H_p, W_p],
                  # Optional:
                  'token_idx': LongTensor[K]
                }
              }
            }
          }
        }
      }
    }

    Reading example:
      blob = torch.load(save_path, map_location='cpu')
      attn = blob['timestep'][t]['layers'][layer]['insts'][inst]['attn_map']
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Load or initialize
    if os.path.isfile(save_path):
        blob = torch.load(save_path, map_location='cpu')
    else:
        blob = {'meta': {'T': T, 'H_p': H_p, 'W_p': W_p}, 'timestep': {}}

    # Meta consistency
    meta = blob.setdefault('meta', {})
    assert meta.get('T', T) == T and meta.get('H_p', H_p) == H_p and meta.get('W_p', W_p) == W_p, \
        f"Meta mismatch: existing={meta}, new=(T={T}, H_p={H_p}, W_p={W_p})"

    timesteps = blob.setdefault('timestep', {})
    ts_slot = timesteps.setdefault(int(progress_id), {})
    layers_slot = ts_slot.setdefault('layers', {})

    # Iterate through layers and write
    for layer_id in sorted(attn_map_history.keys()):
        attn_map = attn_map_history[layer_id][0]  # Tensor(B, N, S, L)
        assert attn_map.dim() == 4, f"attn_map 4D expected, got {attn_map.shape}"

        B, N, S, L = attn_map.shape
        assert S == T * H_p * W_p, f"Expected S={T*H_p*W_p}, got {S}"

        layer_slot = layers_slot.setdefault(int(layer_id), {})
        insts_slot = layer_slot.setdefault('insts', {})

        for n in range(N):
            attn_per_inst = attn_map[0, n]           # (S, L)
            token_mask = (attn_per_inst.sum(dim=0) > 0)
            if token_mask.any():
                attn_valid = attn_per_inst[:, token_mask]    # (S, K)
                attn_avg = attn_valid.mean(dim=1)            # (S,)
                token_idx = torch.nonzero(token_mask, as_tuple=False).view(-1)
            else:
                attn_avg = attn_per_inst.mean(dim=1)         # (S,)
                token_idx = torch.arange(L) if store_token_idx else None

            attn_grid = attn_avg.view(T, H_p, W_p).to(dtype=torch.float32, device='cpu').contiguous()

            slot = {'attn_map': attn_grid}
            if store_token_idx and token_idx is not None:
                slot['token_idx'] = token_idx.to(dtype=torch.long, device='cpu')

            insts_slot[int(n)] = slot

    torch.save(blob, save_path)
    print(f"[OK] Saved attention maps → {save_path}  (root key: 'timestep')")


# def save_attention_maps_from_history(attn_map_history, save_dir, progress_id, T=21, H_p=30, W_p=52):
#     os.makedirs(save_dir, exist_ok=True)

#     for layer_id in sorted(attn_map_history.keys()):
#         attn_map = attn_map_history[layer_id][0]  # Tensor(B, N, S, L)

#         B, N, S, L = attn_map.shape
#         assert S == T * H_p * W_p, f"Expected S={T*H_p*W_p}, got {S}"

#         for n in range(N):
#             attn_per_inst = attn_map[0, n]  # (S, L)
#             token_mask = (attn_per_inst.sum(dim=0) > 0)
#             attn_valid = attn_per_inst[:, token_mask]
#             attn_avg = attn_valid.mean(dim=1)  # shape: (S,)
#             attn_grid = attn_avg.view(T, H_p, W_p)  # (T, H_p, W_p)

#             utils.save_attention_heatmap_frames(
#                 attn_tensor=attn_grid,
#                 save_path=os.path.join(save_dir, f"prog{progress_id}_layer{layer_id}_inst{n}"),
#                 tag="attn",
#                 frame_shape=(H_p, W_p),
#             )

def register_attn_map_hooks(model, attn_map_history, target_layers):
    hook_handles = []

    def make_hook(bid):
        def _hook(attn_map):
            attn_map_history[bid].append(attn_map.clone())
        return _hook

    for block_id, block in enumerate(model.blocks):
        if block_id in target_layers:
            if hasattr(block.cross_attn, 'register_attn_map_hook'):
                handle = block.cross_attn.register_attn_map_hook(make_hook(block_id))
                hook_handles.append(handle)
            else:
                print(f"Warning: block.cross_attn at layer {block_id} does not have register_attn_map_hook method")
                # Create a dummy handle that does nothing
                class DummyHandle:
                    def remove(self):
                        pass
                hook_handles.append(DummyHandle())

    return hook_handles

def safe_kl(p, m, eps=1e-8):
    mask = (p > 0) & (m > 0)  # Only calculate parts where P>0 and M>0 to prevent log(0)
    return (p[mask] * (p[mask] / (m[mask] + eps)).log()).sum()



def jsd_loss(attn: torch.Tensor, mask: torch.Tensor, eps=1e-8, use_gaussian: bool = False, save_mask: bool = False) -> torch.Tensor:
    """
    Jensen-Shannon Divergence loss between attn and mask (optionally a Gaussian soft mask).
    
    Args:
        attn (Tensor): Attention map (H, W)
        mask (Tensor): Binary mask (H, W), values in {0, 1}
        eps (float): Small constant to avoid division by zero
        use_gaussian (bool): If True, convert mask to a Gaussian-weighted soft region

    Returns:
        torch.Tensor: Scalar JSD loss
    """
    device = attn.device
    dtype = attn.dtype

    def _gaussian_from_bbox(_mask_2d: torch.Tensor) -> torch.Tensor:
        """Make a Gaussian soft mask from a single (H, W) binary mask via its bounding box ellipse."""
        H, W = _mask_2d.shape
        y_idx, x_idx = torch.nonzero(_mask_2d, as_tuple=True)
        if len(x_idx) == 0:
            return torch.zeros((H, W), device=_mask_2d.device, dtype=_mask_2d.dtype)

        x_min, x_max = x_idx.min(), x_idx.max()
        y_min, y_max = y_idx.min(), y_idx.max()

        cx = (x_min + x_max).to(_mask_2d.dtype) / 2
        cy = (y_min + y_max).to(_mask_2d.dtype) / 2
        # +1 to prevent zero width/height; /4 controls diffusion extent (adjustable as needed)
        sigma_x = (x_max - x_min + 1).to(_mask_2d.dtype) / 4
        sigma_y = (y_max - y_min + 1).to(_mask_2d.dtype) / 4

        yy, xx = torch.meshgrid(
            torch.arange(H, device=_mask_2d.device, dtype=_mask_2d.dtype),
            torch.arange(W, device=_mask_2d.device, dtype=_mask_2d.dtype),
            indexing="ij",
        )
        gauss = torch.exp(-(((xx - cx) ** 2) / (2 * sigma_x ** 2 + eps) +
                            ((yy - cy) ** 2) / (2 * sigma_y ** 2 + eps)))
        return gauss

    if mask.dim() == 2:
        masks = mask.unsqueeze(0)
    elif mask.dim() == 3:
        masks = mask
    else:
        raise ValueError(f"`mask` must be (H, W) or (K, H, W), got shape {mask.shape}")

    if use_gaussian:
        gaussians = []
        for k in range(masks.shape[0]):
            gaussians.append(_gaussian_from_bbox(masks[k]))
        soft_mask = torch.stack(gaussians, dim=0).sum(dim=0)
    else:
        soft_mask = masks.to(dtype).sum(dim=0)
    
    attn_flat = attn.to(dtype).flatten().clamp(min=0)
    mask_flat = soft_mask.to(dtype).flatten().clamp(min=0)


    P = attn_flat / (attn_flat.sum() + eps)
    Q = mask_flat / (mask_flat.sum() + eps)
    M = 0.5 * (P + Q)

    kl_pm = safe_kl(P, M, eps)
    kl_qm = safe_kl(Q, M, eps)
    jsd = 0.5 * (kl_pm + kl_qm)

    return jsd.to(device=device, dtype=dtype)




def center_of_mass(x, h_range, w_range, eps=1.0e-8):
    com_h = (x.sum(dim=2) * h_range).sum(dim=-1) / (x.sum(dim=(1, 2)) + eps)
    com_w = (x.sum(dim=1) * w_range).sum(dim=-1) / (x.sum(dim=(1, 2)) + eps)
    com_h = torch.nan_to_num(com_h, nan=0.0)
    com_w = torch.nan_to_num(com_w, nan=0.0)
    return com_h, com_w

def com_loss(attn_map, attn_map_t1, mask, mask_t1, com_loss_scale=0.01, com_velocity_loss_scale=5):
    device, dtype = attn_map.device, attn_map.dtype
    if attn_map.ndim == 2:
        attn_map = attn_map.unsqueeze(0)        # (1, H, W)
        attn_map_t1 = attn_map_t1.unsqueeze(0)  # (1, H, W)

    mask_sum = mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=device)

    _, H, W = attn_map.shape
    h_range = torch.arange(H, device=device, dtype=dtype)[None]
    w_range = torch.arange(W, device=device, dtype=dtype)[None]

    com_ca_h, com_ca_w = center_of_mass(attn_map, h_range, w_range)
    com_mask_h, com_mask_w = center_of_mass(mask[None], h_range, w_range)

    diff_h = com_ca_h[..., None, None] - com_mask_h
    diff_w = com_ca_w[..., None, None] - com_mask_w
    com_loss_val = (diff_h ** 2 + diff_w ** 2).mean()

    if mask_t1.sum() > 0:
        com_ca_h_t1, com_ca_w_t1 = center_of_mass(attn_map_t1, h_range, w_range)
        com_mask_h_t1, com_mask_w_t1 = center_of_mass(mask_t1[None], h_range, w_range)
        diff_vh = (com_ca_h_t1 - com_ca_h)[..., None, None] - (com_mask_h_t1 - com_mask_h)
        diff_vw = (com_ca_w_t1 - com_ca_w)[..., None, None] - (com_mask_w_t1 - com_mask_w)
        com_velocity_loss = (diff_vh ** 2 + diff_vw ** 2).mean()
    else:
        com_velocity_loss = torch.tensor(0.0, device=attn_map.device)

    return com_loss_val * com_loss_scale + com_velocity_loss * com_loss_scale * com_velocity_loss_scale


def compute_loss(
        attn_map_history, 
        layout, 
        progress_id, 
        stage_1=0,
        save_attnmap=False,
        jsd_loss_weight=0,
        com_loss_weight=0,
        T=21,
        min_token_attn=False,
        save_mask=False,
        ):
    """
    attn_map_history: Dict[layer_id → List[tensor]], tensor: [B, N, S, L]
    layout: List[{"id": int, "masks": List[Tensor(H, W)]}]
    """
    loss_all = []
    H_p, W_p = 30, 52
    H_t, W_t = H_p*5, W_p*5
    # instance_losses = []
    # 1. Iterate through all target layers (ensure consistent key ordering)
    for layer_id in sorted(attn_map_history.keys()):
        attn_maps = attn_map_history[layer_id]  # List[Tensor(B, N, S, L)]

        for attn_map in attn_maps:
            # shape: (B, N, S, L)
            B, N, S, L = attn_map.shape
            assert S == T * H_p * W_p, f"Expected S={T*H_p*W_p}, got {S}"

            # 2. Iterate through each instance n, matching with layout
            layout_ids = [x["id"] for x in layout]

            from collections import defaultdict
            name_groups = defaultdict(list)
            for item in layout:
                name_groups[item["name"]].append(item)

            for n in range(N):
                if n not in layout_ids:
                    continue
                layout_n = next((x for x in layout if x["id"] == n), None)  # {"id": int, "masks": List[Tensor(H, W)]}
                mask_seq = layout_n["masks"]  # List[Tensor(H, W)], 长度 T=21
                same_name_entries = name_groups[layout_n["name"]]

                # 3. attention map: B=1, attention distribution for all query patches: [S, L]
                attn_per_inst = attn_map[0, n]  # shape (S, L)
                token_mask = (attn_per_inst.sum(dim=0) > 0)
                attn_valid = attn_per_inst[:, token_mask]
                if min_token_attn:
                    attn_intersect = attn_valid.min(dim=1).values  # (S,)
                    attn_grid = attn_intersect.view(T, H_p, W_p)  # (21, 30, 52)
                else:
                    attn_avg = attn_valid.mean(dim=1)  # shape: (S,)
                    attn_grid = attn_avg.view(T, H_p, W_p)  # (21, 30, 52)
                # print("attn_grid.shape: ", attn_grid.shape)

                pid = 48
                if save_attnmap:
                    os.makedirs(f"./attn_vis/pid{pid}", exist_ok=True)
                    utils.save_attention_heatmap_frames(
                        attn_tensor=attn_grid,  # (21, 30, 52)
                        save_path=f"./attn_vis/pid{pid}/ts{progress_id}_layer{layer_id}_inst{n}",
                        tag="attn",
                        frame_shape=(H_p, W_p),
                    )

                loss_per_frame = []
                # Iterate through 21 frames, aligning frame by frame with layout mask
                
                T_cal = T
                if stage_1 and progress_id < stage_1:
                    T_cal=1

                for t in range(T_cal):
                    # gt_mask = mask_seq[t]  # shape: (H_orig, W_orig)
                    attn_frame = attn_grid[t]  # (30, 52)

                    # Resize layout mask to (30, 52)
                    # gt_resized = F.interpolate(
                    #     gt_mask[None, None], 
                    #     size=(H_t, W_t), 
                    #     mode="bilinear", 
                    #     align_corners=False
                    # )
                    attn_frame_resized = F.interpolate(
                        attn_frame.unsqueeze(0).unsqueeze(0),   # (1,1,H_p,W_p)
                        size=(H_t, W_t),                    # Target (H_tgt, W_tgt)
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0).squeeze(0)

                    masks_t_resized = []
                    for entry in same_name_entries:
                        gt_mask_t = entry["masks"][t]  # (H_orig, W_orig)
                        gt_resized_t = F.interpolate(
                            gt_mask_t[None, None],
                            size=(H_t, W_t),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze(0).squeeze(0)
                        # Binarize to {0,1} for subsequent Gaussian or direct stacking
                        gt_bin_t = (gt_resized_t > 0.5).to(dtype=attn_grid.dtype, device=attn_grid.device)
                        masks_t_resized.append(gt_bin_t)
                    gt_stack_t = torch.stack(masks_t_resized, dim=0)

                    # gt_resized = (gt_resized > 0.5).to(dtype=attn_frame_resized.dtype, device=attn_frame_resized.device).squeeze()  # (30, 52)

                    if jsd_loss_weight > 0:
                        # jsd_loss_val = jsd_loss(attn_frame_resized, gt_resized, use_gaussian=True) * jsd_loss_weight
                        # print(f"save_mask: {save_mask}")
                        jsd_loss_val = jsd_loss(attn_frame_resized, gt_stack_t, use_gaussian=True, save_mask=save_mask) * jsd_loss_weight
                        if jsd_loss_val.detach().item() > 1e-8:
                            # loss_per_frame.append(jsd_loss_val)
                            loss_all.append(jsd_loss_val)
                
                    if com_loss_weight > 0:
                        if 0 <= t < T - 1:
                            gt_mask_t1 = mask_seq[t + 1]
                            attn_frame_t1 = attn_grid[t + 1]
                            gt_resized_t1 = F.interpolate(
                                gt_mask_t1[None, None],
                                size=(H_t, W_t),
                                mode="bilinear",
                                align_corners=False
                            )
                            attn_frame_resized_t1 = F.interpolate(
                                attn_frame_t1.unsqueeze(0).unsqueeze(0),   # (1,1,H_p,W_p)
                                size=(H_t, W_t),                    # 目标 (H_tgt, W_tgt)
                                mode="bilinear",
                                align_corners=False
                            ).squeeze(0).squeeze(0)
                            gt_resized_t1 = (gt_resized_t1 > 0.5).to(dtype=attn_frame_resized_t1.dtype, device=attn_frame_resized_t1.device).squeeze()
                            com_loss_val = com_loss(attn_frame_resized, attn_frame_resized_t1, gt_stack_t, gt_resized_t1)
                            loss_all.append(com_loss_val)
                # if loss_per_frame:
                    # instance_loss = torch.stack(loss_per_frame).mean() * jsd_loss_weight
                    # instance_losses.append(instance_loss)

    # return torch.stack(instance_losses).max() if instance_losses else torch.tensor(0.0, device=attn_map.device)
    return torch.stack(loss_all).mean() if loss_all else torch.tensor(0.0, device=attn_map.device)

                    
def print_cuda_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{prefix}] CUDA Memory - Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB | Peak: {peak_allocated:.1f} MB")

from peft import LoraConfig, inject_adapter_in_model
def add_lora_to_model(model, target_modules, lora_rank, lora_alpha=None):
    if lora_alpha is None:
        lora_alpha = lora_rank
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
    model = inject_adapter_in_model(lora_config, model)
    return model

def load_lora_weights_into_model(
    model,
    progress_id: int,
    lora_path: str = "./lora_ckpts",
    target_modules=("q", "k", "v", "o", "ffn.0", "ffn.2"),
    lora_rank: int = 32,
):
    """
    Automatically inject LoRA modules and load weights

    Args:
        model: Original model without LoRA injection
        progress_id: Current task ID, used to locate weight path
        ckpt_root: Root directory for storing LoRA weights
        target_modules: Module names that need LoRA injection
        lora_rank: Rank of injected LoRA modules

    Returns:
        Model with LoRA injected and weights loaded
    """
    import os
    import torch

    # Construct path

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"❌ LoRA checkpoint not found at {lora_path}")

    print(f"🔧 Injecting LoRA modules into model...")
    model = add_lora_to_model(model, target_modules=list(target_modules), lora_rank=lora_rank)

    print(f"📦 Loading LoRA weights from: {lora_path}")
    lora_state_dict = torch.load(lora_path, map_location="cpu")

    missing_keys = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            if name in lora_state_dict:
                param.data.copy_(lora_state_dict[name])
            else:
                missing_keys.append(name)

    if missing_keys:
        print(f"⚠️ Warning: {len(missing_keys)} LoRA parameters not found in checkpoint:")
        for key in missing_keys:
            print(f"   - {key}")

    print("✅ LoRA weights successfully loaded into model.")
    return model



def guidance_lora(
    device,
    model_fn,
    *,
    progress_id: int,
    models: dict,
    inputs_shared: dict,
    inputs_posi: dict,
    timestep: torch.Tensor,
    target_modules=("q", "k", "v", "o", "ffn.0", "ffn.2"),
    lora_rank: int = 32,
    max_iter: int = 5,
    lr: float = 1e-3,
    ckpt_root: str = "./lora_ckpts",
    target_layers=(6, 7, 13, 14, 15),
    min_loss_value = 2.0,
    jsd_loss_weight=0,
    com_loss_weight=0,
    save_lora_weight=False,
    stage_1=0,
    T=21,
    save_mask=False,
):
    if inputs_posi.get("insts_prompts") is None:
        raise RuntimeError("inputs_posi['insts_prompts'] is None.")

    with torch.set_grad_enabled(True):
        dit = models["dit"]
        if not any("lora_" in n for n, _ in dit.named_parameters()):
            dit = add_lora_to_model(dit, target_modules=list(target_modules), lora_rank=lora_rank)
            for n, p in dit.named_parameters():
                if "lora_" in n:
                    p.data = p.data.to(device)
            models["dit"] = dit  # Write back
            

        # Freeze backbone, only enable LoRA
        for n, p in dit.named_parameters():
            p.requires_grad = "lora_" in n
        
        if not hasattr(dit, "_lora_optimizer"):
            opt = torch.optim.AdamW(
                [p for p in dit.parameters() if p.requires_grad], lr=lr
            )
            dit._lora_optimizer = opt
        else:
            opt = dit._lora_optimizer
        opt.zero_grad(set_to_none=True)
        
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(max_iter):
            attn_map_history = {bid: [] for bid in target_layers}
            hook_handles = register_attn_map_hooks(models["dit"], attn_map_history, target_layers)

            _ = model_fn(
                enable_guidance=True,
                **models, 
                **inputs_shared, 
                **inputs_posi, 
                timestep=timestep,
                use_gradient_checkpointing=True
                )
            layout = inputs_posi["layout"]
            loss = compute_loss(
                attn_map_history, 
                layout, 
                progress_id,
                jsd_loss_weight=jsd_loss_weight,
                com_loss_weight=com_loss_weight,
                save_attnmap=False,
                T=T,
                save_mask=save_mask,
                # min_token_attn=True,
                # stage_1=2,
                )

            print(f"[prog {progress_id:02d} | iter {iteration:02d}] loss={loss.item():.4f}")

            if loss.item() < min_loss_value:
                del attn_map_history
                for h in hook_handles:
                    h.remove()
                torch.cuda.empty_cache()
                break

            # (c) 反向 & 更新 LoRA
            opt.zero_grad()
            loss.backward()
            opt.step()


            del attn_map_history
            del loss, layout
            gc.collect()
            for h in hook_handles:
                h.remove()
            torch.cuda.empty_cache()
            print_cuda_memory(prefix=f"after iter {iteration}")
    
    if save_lora_weight:                
        ckpt_dir = os.path.join(
            ckpt_root, f"prog_{progress_id:02d}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {n: p.detach().cpu() for n, p in dit.named_parameters() if "lora_" in n},
            os.path.join(ckpt_dir, "lora.pth"),
        )

    return models

def guidance_lvd(
        model_fn, 
        progress_id, 
        models, 
        inputs_shared, 
        inputs_posi, 
        timestep,
        jsd_loss_weight=0,
        com_loss_weight=0,
    ):
    if inputs_posi["insts_prompts"] is None:
        print("error")

    target_layers = [13, 15]
    latents = inputs_shared["latents"].clone().detach()
    max_iter = 3
    lr = 1
    torch.cuda.reset_peak_memory_stats()
    with torch.set_grad_enabled(True):
        for iteration in range(max_iter):
            latents = latents.requires_grad_(True)
            inputs_shared["latents"] = latents
            attn_map_history = {bid: [] for bid in target_layers}
            hook_handles = register_attn_map_hooks(models["dit"], attn_map_history, target_layers)

            _ = model_fn(
                enable_guidance=True,
                **models, 
                **inputs_shared, 
                **inputs_posi, 
                timestep=timestep,
                use_gradient_checkpointing_offload=True
                )
            layout = inputs_posi["layout"]
            loss = compute_loss(
                attn_map_history, 
                layout, 
                progress_id,
                jsd_loss_weight=jsd_loss_weight,
                com_loss_weight=com_loss_weight,
                save_attnmap=False,
                )
            print("loss: ", loss)
            grad_cond = torch.autograd.grad(loss, [latents])[0]


            # print(f"[Debug] grad_cond.norm = {grad_cond.norm().item():.6f}")
            # print(f"[Debug] grad_cond.mean = {grad_cond.mean().item():.6f}, "
            #     f"max = {grad_cond.max().item():.6f}, min = {grad_cond.min().item():.6f}")
            
            latents.requires_grad_(False)
            latents = latents - lr * grad_cond
            del attn_map_history, grad_cond
            for h in hook_handles:
                h.remove()
            torch.cuda.empty_cache()
            if loss.item() < 10.0:
                break
            # print_cuda_memory(prefix=f"after iter {iteration}")
    return latents.detach()

    # process_and_save_attention_maps(
    #     attn_map_history=attn_map_history,
    #     target_layers=target_layers,
    #     concept_prompt=inputs_posi["concept_prompt"],
    #     progress_id=progress_id,
    #     frame_shape=(30, 52),
    # )


def process_and_save_attention_maps(
    attn_map_history: dict,
    target_layers: list,
    concept_prompt: str,
    progress_id: int,
    frame_shape=(30, 52),
    save_root="data/attnmap"
):
    """
    Process and save attention maps visualization and stacking results.

    Args:
    - attn_map_history: dict[block_id][attn_type] -> Tensor[B, T, D]
    - target_layers: List of block ids to process
    - inputs_posi: Input dictionary containing concept_prompt
    - progress_id: Current step progress ID, used for naming output files
    - frame_shape: Shape of each frame (H, W)
    - save_root: Directory path for saving images and tensors
    """
    if concept_prompt is None:
        print("⚠️ Skip attention processing: concept_prompt is None")
        return

    num_attn_types = len(attn_map_history[target_layers[0]])

    for idx in range(num_attn_types):
        all_blocks_tensor = []
        blocks_to_save = []
        for bid in target_layers:
            # Step 1: Save attention visualization for each block separately
            attn_tensor = attn_map_history[bid][idx]  # shape: [1, F*H*W, num_heads]
            avg_attn = attn_tensor[0].mean(dim=-1)    # shape: [F*H*W]

            if bid in blocks_to_save:
                utils.save_attention_heatmap_frames(
                    attn_tensor=avg_attn,
                    save_path=os.path.join(save_root, f"s{progress_id}_{concept_prompt}_b{bid}"),
                    tag=f"attn_{idx}",
                    frame_shape=frame_shape,
                )

            all_blocks_tensor.append(avg_attn)  # For pt saving

        # Step 2: Stack all block attention and save as pt
        result_tensor = torch.stack(all_blocks_tensor, dim=0)  # shape: [num_blocks, F*H*W]
        pt_save_path = os.path.join(save_root, f"s{progress_id}_{concept_prompt}_bstack_attn_{idx}.pt")
        torch.save(result_tensor, pt_save_path)
        print(f"📦 Saved stacked attn map to: {pt_save_path}, shape={result_tensor.shape}")