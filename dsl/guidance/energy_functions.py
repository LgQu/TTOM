import torch
import torch.nn.functional as F
from collections.abc import Iterable
import warnings
from typing import List
import dsl.utils as utils
import inflect

p = inflect.engine()
from sklearn.cluster import KMeans
import numpy as np

def filter_attn_with_soft_threshold(attn_map_2d, n_clusters=2):
    """
    Differentiable version of filter_attn_with_kmeans:
    - attn_map_2d: (H, W) tensor
    - n_clusters: ignored here, kept for compatibility
    Returns:
    - (H, W) filtered attention map
    """
    H, W = attn_map_2d.shape

    attn_min = attn_map_2d.min()
    attn_max = attn_map_2d.max()
    attn_norm = (attn_map_2d - attn_min) / (attn_max - attn_min + 1e-8)

    threshold = 0.5

    sharpness = 10.0
    soft_mask = torch.sigmoid((attn_norm - threshold) * sharpness)

    attn_map_filtered = attn_map_2d * soft_mask

    return attn_map_filtered

def ciou_based_loss(attn_map, mask, eps=1e-2, ciou_based_loss_weight=10):
    import torch

    b, H, W = attn_map.shape
    losses = []

    for i in range(b):
        attn = attn_map[i]  # (H, W)
        # print(f"[{i}] Raw attention min: {attn.min().item()}, max: {attn.max().item()}", flush=True)

        # Normalize attention map to range [0, 1]
        attn = (attn - attn.min()) / (attn.max() - attn.min() + eps)
        # print(f"[{i}] Normalized attention - min: {attn.min().item()}, max: {attn.max().item()}", flush=True)

        threshold = 0.5
        attn_bin = attn > threshold
        activated_ratio = attn_bin.sum().float() / (H * W)

        # Fallback to Top-k if activated_ratio is too low
        if activated_ratio < 0.01:
            flat_attn = attn.flatten()
            k = max(int(flat_attn.numel() * 0.01), 1)  # top 1%
            threshold = torch.kthvalue(flat_attn, flat_attn.numel() - k + 1).values
            attn_bin = attn >= threshold
            # print(f"[{i}] Fallback to Top-k threshold: {threshold.item():.6f} (activated_ratio={activated_ratio:.4f})", flush=True)
            # print(f"[{i}] Using static threshold {threshold} (activated_ratio={activated_ratio:.4f})", flush=True)

        if attn_bin.sum() == 0:
            # print(f"[{i}] No activation found in attention map.", flush=True)
            continue

        attn_coords = attn_bin.nonzero(as_tuple=False)
        y1_a, x1_a = attn_coords.min(dim=0).values
        y2_a, x2_a = attn_coords.max(dim=0).values
        # print(f"[{i}] Attn bbox: ({x1_a.item()}, {y1_a.item()}) to ({x2_a.item()}, {y2_a.item()})", flush=True)

        mask_coords = (mask > 0).nonzero(as_tuple=False)
        if mask_coords.numel() == 0:
            # print(f"[{i}] No mask region found.", flush=True)
            continue

        y1_m, x1_m = mask_coords.min(dim=0).values
        y2_m, x2_m = mask_coords.max(dim=0).values
        # print(f"[{i}] Mask bbox: ({x1_m.item()}, {y1_m.item()}) to ({x2_m.item()}, {y2_m.item()})", flush=True)

        def to_cxcywh(x1, y1, x2, y2, eps):
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            return cx, cy, w + eps, h + eps

        cx_a, cy_a, w_a, h_a = to_cxcywh(x1_a, y1_a, x2_a, y2_a, eps)
        cx_m, cy_m, w_m, h_m = to_cxcywh(x1_m, y1_m, x2_m, y2_m, eps)

        # print(f"[{i}] w_a: {w_a.item()}, h_a: {h_a.item()}, w_m: {w_m.item()}, h_m: {h_m.item()}", flush=True)

        inter_w = torch.clamp(torch.min(x2_a, x2_m) - torch.max(x1_a, x1_m), min=0)
        inter_h = torch.clamp(torch.min(y2_a, y2_m) - torch.max(y1_a, y1_m), min=0)
        inter_area = inter_w * inter_h

        area_a = (x2_a - x1_a) * (y2_a - y1_a) + eps
        area_m = (x2_m - x1_m) * (y2_m - y1_m) + eps
        union_area = area_a + area_m - inter_area + eps
        iou = inter_area / union_area
        # print(f"[{i}] IoU: {iou.item()}", flush=True)

        center_dist = (cx_a - cx_m) ** 2 + (cy_a - cy_m) ** 2
        enclose_x1 = torch.min(x1_a, x1_m)
        enclose_y1 = torch.min(y1_a, y1_m)
        enclose_x2 = torch.max(x2_a, x2_m)
        enclose_y2 = torch.max(y2_a, y2_m)
        c_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

        diou_term = center_dist / c_diag
        # print(f"[{i}] center_dist: {center_dist.item()}, c_diag: {c_diag.item()}, DIoU term: {diou_term.item()}", flush=True)

        v = (4 / (torch.pi ** 2)) * (torch.atan(w_m / h_m) - torch.atan(w_a / h_a)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        # print(f"[{i}] v: {v.item()}, alpha: {alpha.item()}", flush=True)

        ciou = iou - diou_term - alpha * v
        loss = 1 - ciou
        # print(f"[{i}] CIoU: {ciou.item()}, Loss: {loss.item()}", flush=True)
        losses.append(loss)

    if len(losses) == 0:
        # print("No valid samples found, returning zero loss.", flush=True)
        return torch.tensor(0.0, device=attn_map.device)

    total_loss = ciou_based_loss_weight * torch.stack(losses).mean()
    # print(f"Total CIoU Loss: {total_loss.item()}", flush=True)
    return total_loss


def ratio_based_loss(attn_map, mask, eps=1.0e-2, ratio_based_loss_weight=10):
    # Enforces the attention to be within the mask only. Does not enforce within-mask distribution.
    b = attn_map.shape[0]
    warnings.warn(
        "Using ratio-based loss, which is deprecated. Max-based loss is recommended. The scale may be different."
    )
    activation_value = (attn_map * mask).reshape(b, -1).sum(
        dim=-1
    ) / (attn_map.reshape(b, -1).sum(dim=-1) + eps)
    loss = torch.mean((1 - activation_value) ** 2) * ratio_based_loss_weight
    return loss

def max_based_loss(attn_map_2d, mask, min_topk_size=16, fg_top_p=0.2, bg_top_p=0.2, fg_weight=1.0, bg_weight=1.0, eps=1.0e-2):
    # Compute the number of top elements for foreground and background
    k_fg = (mask.sum() * fg_top_p).long().clamp_(min=min_topk_size)
    k_bg = ((1 - mask).sum() * bg_top_p).long().clamp_(min=min_topk_size)

    mask_1d = mask.view(1, -1)

    try:
        # Compute foreground loss using top k values
        fg_values = (attn_map_2d * mask_1d).topk(k=k_fg).values
        fg_loss = (1 - fg_values.mean(dim=1)).sum(dim=0)
        # Compute background loss using top k values
        bg_values = (attn_map_2d * (1 - mask_1d)).topk(k=k_bg).values
        bg_loss = bg_values.mean(dim=1).sum(dim=0)
        loss = fg_loss * fg_weight + bg_loss * bg_weight
    except Exception as e:
        print("Exception in max_based_loss:", e)
        loss = torch.tensor(float('nan'))

    # If loss is NaN, print all input parameters for debugging
    if torch.isnan(loss):
        print("NaN detected in max_based_loss!")
        print("attn_map_2d:", attn_map_2d)
        print("mask:", mask)
        print("k_fg:", k_fg)
        print("k_bg:", k_bg)
    return loss

def jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    p = p.view(-1)
    q = q.view(-1)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    
    # Avoid log(0)
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    m = 0.5 * (p + q)
    m = torch.clamp(m, min=eps)
    
    jsd_val = 0.5 * torch.sum(p * torch.log(p / m)) + \
            0.5 * torch.sum(q * torch.log(q / m))
    return jsd_val

def ce_based_loss(
    attn_map_2d, 
    mask, 
    min_topk_size=16, 
    fg_top_p=0.2, 
    bg_top_p=0.2, 
    fg_weight=1.0, 
    bg_weight=1.0, 
    eps=1.0e-2
    ):
    # Take the top k over spatial dimension, and then take the sum over heads dim
    # The mean is over k_fg and k_bg dimension, so we don't need to sum and divide on our own.
    # clamp is for numerical stability
    k_fg = (mask.sum() * fg_top_p).long().clamp_(min=min_topk_size)
    k_bg = ((1 - mask).sum() * bg_top_p).long().clamp_(min=min_topk_size)

    attn_map_2d = torch.clamp(attn_map_2d, min=eps, max=1 - eps)
    mask_1d = mask.view(1, -1)

    fg_loss = (
        -torch.log(
            torch.clamp(
                (mask_1d * attn_map_2d).topk(k=k_fg).values, min=eps
            )
        ).mean(dim=1).sum(dim=0)
    )

    bg_loss = -torch.log(1 - ((1 - mask_1d) * attn_map_2d).topk(k=k_bg).values.mean(dim=1)).sum(dim=0)

    if torch.isinf(fg_loss) or torch.isinf(bg_loss):
        warnings.warn(
            f"Encountered inf loss. fg_loss: {fg_loss}, bg_loss: {bg_loss}"
        )

    return fg_loss * fg_weight + bg_loss * bg_weight

def attn_sync_loss(attn_map, bboxes, object_positions, object_number):
    # Todo
    return 0

def boxdiff_loss(attn_map, mask, boxdiff_normed=True, boxdiff_L=1, boxdiff_loss_scale=0.0):
    b, H, W = attn_map.shape
    # attn_map: (b, H, W)
    # attn_map_max_x: (b, W)
    # attn_map_max_y: (b, H)
    attn_map_max_x = attn_map.max(dim=1).values.to(torch.float32)
    attn_map_max_y = attn_map.max(dim=2).values.to(torch.float32)
    # mask: (H, W)
    # mask_max_x: (1, W)
    # mask_max_y: (1, H)
    mask_max_x = mask[None].max(dim=1).values
    mask_max_y = mask[None].max(dim=2).values

    # corner_mask_x: (1, W)
    # corner_mask_y: (1, H)
    indices = torch.nonzero(mask, as_tuple=True)
    y_min, y_max = indices[0].min().item(), indices[0].max().item() + 1
    x_min, x_max = indices[1].min().item(), indices[1].max().item() + 1
    corner_mask_x = torch.zeros(size=(1, W), device=attn_map.device)
    corner_mask_x[:, max(0, x_min - boxdiff_L):min(x_min + boxdiff_L + 1, W)] = 1.0
    corner_mask_x[:, max(x_max - boxdiff_L, 0):min(x_max + boxdiff_L + 1, W)] = 1.0
    corner_mask_y = torch.zeros(size=(1, H), device=attn_map.device)
    corner_mask_y[:, max(y_min - boxdiff_L, 0):min(y_min + boxdiff_L + 1, H)] = 1.0
    corner_mask_y[:, max(y_max - boxdiff_L, 0):min(y_max + boxdiff_L + 1, H)] = 1.0
    cc_loss_x = (attn_map_max_x - mask_max_x).abs() * corner_mask_x
    cc_loss_y = (attn_map_max_y - mask_max_y).abs() * corner_mask_y
    if boxdiff_normed:
        cc_loss = cc_loss_x.mean() + cc_loss_y.mean()
    else:
        cc_loss = cc_loss_x.sum() + cc_loss_y.sum()
    return cc_loss * boxdiff_loss_scale

def center_of_mass(x, h_range, w_range, eps=1.0e-8):
    com_h = (x.sum(dim=2) * h_range).sum(dim=-1) / (x.sum(dim=(1, 2)) + eps)
    com_w = (x.sum(dim=1) * w_range).sum(dim=-1) / (x.sum(dim=(1, 2)) + eps)
    com_h = torch.nan_to_num(com_h, nan=0.0)
    com_w = torch.nan_to_num(com_w, nan=0.0)
    return com_h, com_w


def com_loss(attn_map, attn_map_t1, mask, mask_t1, com_loss_scale=0.01, com_velocity_loss_scale=5):
    # Calculate the sum of mask and return 0 if it's all zeros
    mask_sum = mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=attn_map.device)
    
    b, H, W = attn_map.shape
    h_range = torch.arange(H, device=attn_map.device, dtype=attn_map.dtype)[None]
    w_range = torch.arange(W, device=attn_map.device, dtype=attn_map.dtype)[None]

    # Compute center of mass for the attention map and mask at time t0
    com_ca_h, com_ca_w = center_of_mass(attn_map.to(torch.float32), h_range, w_range)
    com_mask_h, com_mask_w = center_of_mass(mask[None].to(torch.float32), h_range, w_range)


    diff_h = com_ca_h[..., None, None] - com_mask_h
    diff_w = com_ca_w[..., None, None] - com_mask_w
    com_loss_val = (diff_h ** 2 + diff_w ** 2).mean()

    # Compute velocity loss for time t1 if mask_t1 has non-zero elements
    if mask_t1.sum() > 0:
        mask_t1_sum = mask_t1.sum()
        com_ca_h_t1, com_ca_w_t1 = center_of_mass(attn_map_t1.to(torch.float32), h_range, w_range)
        com_mask_h_t1, com_mask_w_t1 = center_of_mass(mask_t1[None].to(torch.float32), h_range, w_range)
        diff_vh_t0_t1 = (com_ca_h_t1 - com_ca_h)[..., None, None] - (com_mask_h_t1 - com_mask_h)
        diff_vw_t0_t1 = (com_ca_w_t1 - com_ca_w)[..., None, None] - (com_mask_w_t1 - com_mask_w)
        com_velocity_loss = (diff_vh_t0_t1 ** 2 + diff_vw_t0_t1 ** 2).mean()
    else:
        mask_t1_sum = torch.tensor(0.0, device=attn_map.device)
        com_velocity_loss = torch.tensor(0.0, device=attn_map.device)

    loss = com_loss_val * com_loss_scale + com_velocity_loss * com_loss_scale * com_velocity_loss_scale

    # Only output intermediate values when NaN is detected
    if torch.isnan(loss):
        print("NaN detected in com_loss!")
        print("mask sum:", mask_sum)
        print("attn_map shape:", attn_map.shape)
        print("h_range:", h_range)
        print("w_range:", w_range)
        print("Center of mass of attn_map:", com_ca_h, com_ca_w)
        print("Center of mass of mask:", com_mask_h, com_mask_w)
        print("diff_h:", diff_h)
        print("diff_w:", diff_w)
        print("com_loss_val:", com_loss_val)
        print("mask_t1 sum:", mask_t1_sum)
        if mask_t1_sum > 0:
            print("Center of mass of attn_map_t1:", com_ca_h_t1, com_ca_w_t1)
            print("Center of mass of mask_t1:", com_mask_h_t1, com_mask_w_t1)
            print("diff_vh_t0_t1:", diff_vh_t0_t1)
            print("diff_vw_t0_t1:", diff_vw_t0_t1)
            print("com_velocity_loss:", com_velocity_loss)
        else:
            print("mask_t1 is zero, com_velocity_loss is 0")
        print("Final loss:", loss)

    # print(f"com_loss: {loss}")
    return loss


def get_com(map: torch.Tensor) -> torch.Tensor:
    H, W = map.shape
    device = map.device
    y_coord = torch.linspace(0, 1, steps=H, device=device).view(H, 1).expand(H, W)
    x_coord = torch.linspace(0, 1, steps=W, device=device).view(1, W).expand(H, W)
    map_sum = map.sum() + 1e-8
    map_cx = (x_coord * map).sum() / map_sum
    map_cy = (y_coord * map).sum() / map_sum
    map_center = torch.stack([map_cx, map_cy])
    return map_center



def convert_positions_to_global_ranks(object_positions_id_dict: dict[str, list[int]]) -> dict[str, list[int]]:
    all_positions = []
    for obj_key, positions in object_positions_id_dict.items():
        for idx in positions:
            all_positions.append(idx)
    
    sorted_positions = sorted(all_positions)
    position_to_rank = {pos: rank for rank, pos in enumerate(sorted_positions)}

    result = {}
    for obj_key, positions in object_positions_id_dict.items():
        result[obj_key] = [position_to_rank[pos] for pos in positions]
    
    return result

def region_loss(attn_map_obj, attn_map_obj_t1, mask, mask_t1, region_loss_scale, speed_loss_scale=50):
    attn_map_obj = attn_map_obj.mean(dim=0)
    attn_map_obj_t1 = attn_map_obj_t1.mean(dim=0)
    jsd_loss = jsd(attn_map_obj, mask)
    attn_com = get_com(attn_map_obj)
    attn_com_t1 = get_com(attn_map_obj_t1)
    mask_com = get_com(mask)
    mask_com_t1 = get_com(mask_t1)
    speed_loss = torch.norm((attn_com_t1 - attn_com) - (mask_com_t1 - mask_com), p=2) * speed_loss_scale
    return jsd_loss, speed_loss

def info_entropy_loss(attn_map_obj):
    attn = attn_map_obj.mean(dim=0)
    attn = attn / (attn.sum() + 1e-8)
    attn = attn.view(-1)
    attn = torch.clamp(attn, min=1e-8)
    entropy = -torch.sum(attn * torch.log(attn))
    return entropy

def add_ca_loss_per_attn_map_to_loss(
    loss,
    attn_map,
    upsample_scale=1,

    # object information
    object_number=0,
    bboxes=None,
    object_positions_id_dict=None,

    # loss function selection
    use_ciou_based_loss=True,
    use_ratio_based_loss=False,
    use_max_based_loss=False,
    use_ce_based_loss=False,

    # loss function parameters
    fg_top_p=0.2,
    bg_top_p=0.2,
    fg_weight=1.0,
    bg_weight=1.0,
    eps=1.0e-2,
    attn_sync_weight=0.0,
    boxdiff_loss_scale=0.0,
    boxdiff_normed=True,
    boxdiff_L=1,
    com_loss_scale=0.0,
    region_loss_scale=0.0,
    info_entropy_loss_scale=0.0,
    set_cluster_norm=False,

    # control parameters
    index=None,
    verbose=False,
    exclude_bg_heads=False,

    # smoothing parameters  
    smooth_attn=False,
    kernel_size=3,
    sigma=0.5,

    # normalization parameters
    attn_renorm=False,
    renorm_scale=2.0,
):
    """
    fg_top_p, bg_top_p, fg_weight, and bg_weight are only used with max-based loss

    NOTE: default loss is max based now.

    `index` (timestep) is for debugging.
    """

    assert not exclude_bg_heads, "exclude_bg_heads is deprecated"

    # Example attn shape: [30, 13, 30, 45, 64]
    # b: the number of heads
    # n_f: the number of latent frames
    # H: the height of the attention map
    # W: the width of the attention map
    # tokens_num: the number of tokens (64, but only idx in object_positions_id_list are used)
    b, n_f, H_attn, W_attn, tokens_num = attn_map.shape
    # print(f"[0] attn_map shape: {attn_map.shape}")
    # print("[1] tokens_num: ", tokens_num)
    H = int(H_attn * upsample_scale)
    W = int(W_attn * upsample_scale)
    device = attn_map.device

    obj_token_id_dict = convert_positions_to_global_ranks(object_positions_id_dict)
    # print("[2] obj_token_id_dict: ", obj_token_id_dict)
    max_obj_info_entropy_loss = torch.tensor(0.0).cuda()
    for obj_idx in range(object_number):
        obj_loss = 0
        obj_boxes = bboxes[obj_idx]
        assert (
            n_f == len(obj_boxes)
        ), f"Number of frames {n_f} mismatches with number of frames in box condition {len(obj_boxes)}"
        

        token_id_list = obj_token_id_dict[str(obj_idx)]
        # print(f"[3] obj_{obj_idx} token_id_list: ", token_id_list)
        max_frame_info_entropy_loss = torch.tensor(0.0).cuda()
        for frame_id, frame_boxes in enumerate(obj_boxes):
            mask = torch.zeros(size=(H, W), device=device)
            # t1 means the next frame
            mask_t1 = torch.zeros(size=(H, W), device=device)
            frame_id_t1 = min(frame_id + 1, len(obj_boxes) - 1)
            frame_boxes_t1 = obj_boxes[frame_id_t1]

            # for obj_box in frame_boxes:
            x1, y1, x2, y2 = utils.scale_proportion(frame_boxes, H, W)
            # print("- mask [x1,y1,x2,y2]:", [x1, y1, x2, y2])
            mask[y1:y2, x1:x2] = 1

            # for obj_box in frame_boxes_t1:
            x1_t1, y1_t1, x2_t1, y2_t1 = utils.scale_proportion(frame_boxes_t1, H, W)
            mask_t1[y1_t1:y2_t1, x1_t1:x2_t1] = 1
            
            # print(f"[4] attn_map shape: {attn_map.shape}")
            attn_map_obj = attn_map[:, frame_id, :, :, token_id_list]
            attn_map_obj = attn_map_obj.mean(dim=-1).view(b, 1, H_attn, W_attn)
            # print(f"[5] attn_map_obj after mean shape: {attn_map_obj.shape}")
            attn_map_obj_t1 = attn_map[:, frame_id_t1, :, :, token_id_list]
            attn_map_obj_t1 = attn_map_obj_t1.mean(dim=-1).view(b, 1, H_attn, W_attn)

            obj_phrase_loss = 0

            if upsample_scale != 1:
                attn_map_obj = F.interpolate(attn_map_obj, size=(H, W), mode="bilinear", align_corners=False).view(b, H, W)
                attn_map_obj_2d = attn_map_obj.view(b, -1)
                # print(f"[6] attn_map_obj shape after upsample: {attn_map_obj.shape}")

                attn_map_obj_t1 = F.interpolate(attn_map_obj_t1, size=(H, W), mode="bilinear", align_corners=False).view(b, H, W)

            if set_cluster_norm:
                # print("set_cluster_norm: ", set_cluster_norm)
                attn_map_mean = attn_map_obj.mean(dim=0)  # (H, W)
                attn_map_t1_mean = attn_map_obj_t1.mean(dim=0)  # (H, W)
                attn_map_obj = filter_attn_with_soft_threshold(attn_map_mean).unsqueeze(0)
                attn_map_obj_t1 = filter_attn_with_soft_threshold(attn_map_t1_mean).unsqueeze(0)

            if use_ciou_based_loss:
                obj_phrase_loss += ciou_based_loss(attn_map_obj, mask)
            elif use_ratio_based_loss:
                obj_phrase_loss += ratio_based_loss(attn_map_obj, mask)
            elif use_max_based_loss:
                obj_phrase_loss += max_based_loss(
                    attn_map_obj_2d, 
                    mask,
                    min_topk_size=upsample_scale * upsample_scale,
                    fg_top_p=fg_top_p,
                    bg_top_p=bg_top_p,
                    fg_weight=fg_weight,
                    bg_weight=bg_weight,
                    eps=eps,
                )
            elif use_ce_based_loss:
                obj_phrase_loss += ce_based_loss(
                    attn_map_obj_2d, 
                    mask,
                    min_topk_size=upsample_scale * upsample_scale,
                    fg_top_p=fg_top_p,
                    bg_top_p=bg_top_p,
                    fg_weight=fg_weight,
                    bg_weight=bg_weight,
                    eps=eps,
                )
            # else:
                # raise ValueError("Unknown loss: no loss selected")

            if attn_sync_weight != 0.0:
                obj_phrase_loss += attn_sync_loss(attn_map_obj, mask, mask_t1)

            if boxdiff_loss_scale > 0:
                obj_phrase_loss += boxdiff_loss(attn_map_obj, mask, boxdiff_normed, boxdiff_L)

            if com_loss_scale > 0:
                obj_phrase_loss += com_loss(attn_map_obj, attn_map_obj_t1, mask, mask_t1, com_loss_scale)

            if info_entropy_loss_scale > 0:
                entropy = info_entropy_loss(attn_map_obj) * info_entropy_loss_scale
                max_frame_info_entropy_loss = torch.max(max_frame_info_entropy_loss, entropy)


            if region_loss_scale > 0:
                jsd_loss, speed_loss= region_loss(attn_map_obj, attn_map_obj_t1, mask, mask_t1, region_loss_scale)
                print(f"[region_loss] jsd_loss: {jsd_loss}; speed_loss: {speed_loss}; region_loss_scale: {region_loss_scale}")
                obj_phrase_loss += (jsd_loss + speed_loss) * region_loss_scale
            
            
            obj_loss += obj_phrase_loss
        print(f"object_idx: {obj_idx}, obj_loss: {obj_loss/object_number}")
        loss += obj_loss / object_number

    if info_entropy_loss_scale > 0:
        max_obj_info_entropy_loss = torch.max(max_obj_info_entropy_loss, max_frame_info_entropy_loss)
        print(f"max_frame_info_entropy_loss: {max_frame_info_entropy_loss}")
        loss += max_frame_info_entropy_loss

    return loss


def compute_ca_lossv3(
    saved_attn,
    bboxes,
    object_positions_id_dict,
    latent_frames,
    guidance_attn_keys,
    index=None,
    verbose=False,
    set_cluster_norm=False,
    **kwargs,
):
    loss = torch.tensor(0).float().cuda()
    object_number = len(bboxes)
    if object_number == 0:
        return loss
    # print("object_number: ", object_number)
    for attn_key in guidance_attn_keys:
        print(f"attn_layer: {attn_key}")
        # Since in this stage, we only consider cond_embeddings, so we only need the first element of the saved_attn (meaning of [0])
        attn_map_integrated = saved_attn[attn_key][0]
        if not attn_map_integrated.is_cuda:
            attn_map_integrated = attn_map_integrated.cuda()
        object_positions_id_list = sorted(set(token_id for obj_list in object_positions_id_dict.values() for token_id in obj_list))
        tokens_num = len(object_positions_id_list)
        # print(f"[1] tokens_num: {tokens_num}")
        # Example dimension: [17550, 64]
        head, _, _ = attn_map_integrated.shape
        height = 30
        width = 45
        attn_map = attn_map_integrated[:, :, :tokens_num]
        attn_map = attn_map.view(head, latent_frames, height, width, tokens_num)

        loss = add_ca_loss_per_attn_map_to_loss(
            loss=loss,
            attn_map=attn_map,
            upsample_scale=4,
            object_number=object_number,
            bboxes=bboxes,
            index=index,
            verbose=verbose,
            object_positions_id_dict=object_positions_id_dict,
            set_cluster_norm=set_cluster_norm,
            **kwargs,
        )

    num_attn = len(guidance_attn_keys)

    if num_attn > 0:
        loss = loss / (object_number * num_attn)

    return loss
