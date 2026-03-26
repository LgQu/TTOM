import torch
from dsl.guidance.energy_functions import convert_positions_to_global_ranks
import dsl.utils as utils
import torch.nn.functional as F
from torch.nn.functional import softmax

def amf_loss(
    saved_attn,                 # Dict[layer_name] -> tensor [head, 17550, 64]
    amf_map,                    # shape: [n_frames * n_frames, h * w, 2]
    bboxes,                     # List[List[Box]]
    object_positions_id_dict,   # Dict[str(obj_idx)] -> List[token_ids]
    guidance_attn_keys,         # default: cogvideox: [20, 21]
    index,                      
    latent_frames: int = 13,
    latent_H: int = 30,
    latent_W: int = 45,
    verbose: bool = False,
    upsample_scale: float = 1.0
):
    loss = torch.tensor(0.0).cuda()
    device = loss.device
    object_number = len(bboxes)
    if object_number == 0:
        return loss

    H = int(latent_H * upsample_scale)
    W = int(latent_W * upsample_scale)
    key = sorted(guidance_attn_keys)[-1]
    amf_map = amf_map[key]


    object_positions_id_list = sorted(set(token_id for obj_list in object_positions_id_dict.values() for token_id in obj_list))
    tokens_num = len(object_positions_id_list)
    
    attn_list = [saved_attn[k][0] for k in guidance_attn_keys]
    object_attn_all = torch.stack(attn_list, dim=0).mean(dim=0)[..., :tokens_num]

    # def compute_global_flow_loss():
    #     device = amf_map.device
    #     loss = torch.tensor(0.).to(device)
    #     count = 0

    #     for t in range(latent_frames - 1):
    #         gt_flow_map = torch.zeros((H, W, 2), device=device)
    #         for obj_boxes in reversed(bboxes):
    #             if t + 1 >= len(obj_boxes):
    #                 continue
    #             box0 = obj_boxes[t]
    #             box1 = obj_boxes[t + 1]
    #             if box0 is None or box1 is None:
    #                 continue

    #             flow = box_center(box1) - box_center(box0)

    #             mask = torch.zeros((H, W), device=device)
    #             x1, y1, x2, y2 = utils.scale_proportion(box0, H, W)
    #             mask[y1:y2, x1:x2] = 1

    #             gt_flow_map[mask.bool()] = flow

    #         # predicted flow
    #         amf_idx = t * (latent_frames + 1) + 1
    #         pred_flow_map = amf_map[amf_idx].view(H, W, 2)

    #         flow_mask = (gt_flow_map.norm(dim=-1) > 0)

    #         if flow_mask.sum() > 0:
    #             frame_loss = F.mse_loss(
    #                 pred_flow_map,  # shape: [N, 2]
    #                 gt_flow_map,    # shape: [N, 2]
    #                 reduction="mean"
    #             )
    #         else:
    #             frame_loss = torch.tensor(0.0, device=device)
    #         print(f"[global] frame {t} -> {t+1} - frame_flow_loss: {frame_loss}")
    #         loss += frame_loss
    #         count += 1

    #     if count > 0:
    #         loss = loss / count
    #     return loss

    def subtract_token_mean_per_frame(tensor, token_ids):
        # all_tokens = list(range(tokens_num))
        # other_tokens = [t for t in all_tokens if t not in token_ids]
        token_map = tensor[:, :, token_ids].mean(dim=(0, 2))
        # other_map = tensor[:, :, other_tokens].mean(dim=(0, 2))       
        # adjusted = token_map - other_map
        # adjusted = torch.clamp(adjusted, min=0.0)
        adjusted = torch.clamp(token_map, min=0.0)
        return adjusted.view(latent_frames, latent_H, latent_W)

    def box_center(box):
        x, y, w, h = box
        return torch.tensor([x + w / 2, y + h / 2], device=device)

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
    
    def get_mask(box) -> torch.Tensor:
        mask = torch.zeros(size=(H, W), device=device)
        x1, y1, x2, y2 = utils.scale_proportion(box, H, W)
        mask[y1:y2, x1:x2] = 1
        return mask
    
    def attn_topk_selection(attn, mask, rate=5) -> torch.Tensor:
        attn = attn / (attn.sum() + 1e-8)
        mask_count = int(mask.sum().item())
        k = mask_count // rate + 1
        attn_flat = attn.view(-1)
        topk_values, _ = torch.topk(attn_flat, k=k, sorted=False)
        threshold = topk_values.min()
        sharpness = 50
        attn_mask = torch.sigmoid((attn_flat - threshold) * sharpness)
        attn_flat = attn_flat * attn_mask
        attn = attn_flat.view(H, W)
        return attn / (attn.sum() + 1e-8)

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
    
    def info_entropy_loss(attn):
        attn = attn / (attn.sum() + 1e-8)
        attn = attn.view(-1)
        attn = torch.clamp(attn, min=1e-8)
        entropy = -torch.sum(attn * torch.log(attn))
        return entropy

    max_obj_info_entropy_loss = torch.tensor(0.0).cuda()
    object_positions_id_dict = convert_positions_to_global_ranks(object_positions_id_dict)
    for obj_idx, obj_boxes in enumerate(bboxes):
        if str(obj_idx) not in object_positions_id_dict:
            continue
        obj_loss = torch.tensor(0.0).cuda()
        token_ids = object_positions_id_dict[str(obj_idx)]
        token_map = subtract_token_mean_per_frame(object_attn_all, token_ids)

        max_frame_info_entropy_loss = torch.tensor(0.0).cuda()
        for t in range(latent_frames - 1):
            if t + 1 >= len(obj_boxes):
                continue
            box0 = obj_boxes[t]
            box1 = obj_boxes[t + 1]
            if box0 is None or box1 is None:
                continue

            mask = get_mask(box=box0)
            mask_t1 = get_mask(box=box1)
            attn = token_map[t]

            entropy = info_entropy_loss(attn)
            max_frame_info_entropy_loss = torch.max(max_frame_info_entropy_loss, entropy)

            attn_t1 = token_map[t+1]
            attn_topk = attn_topk_selection(attn=attn, mask=mask)
            attn_t1_topk = attn_topk_selection(attn=attn_t1, mask=mask_t1)

            print(f"[debug] attn max after masking: {attn_topk.max().item():.6f}")


            region_loss = jsd(attn_topk, mask)

            
            # com_loss = torch.norm(attn_com - mask_com, p=2)

            # print(f"[1] obj {obj_idx} - frame {t} - com_loss: {com_loss}")

            # inside_sum = (mask * attn).sum()
            # outside_sum = ((1.0 - mask) * attn).sum()
            # inside_weight = mask.sum() + 1e-8
            # outside_weight = (1.0 - mask).sum()/5.0 + 1e-8
            # inside_loss = (1 - inside_sum) / inside_weight
            # outside_loss = outside_sum / outside_weight

            # print(f"[1] obj {obj_idx} - frame {t} - inside_loss: {inside_loss}")
            # print(f"[1] obj {obj_idx} - frame {t} - outside_loss: {outside_loss}")
            
            # region_loss = com_loss + inside_loss + outside_loss
            print(f"[1] obj {obj_idx} - frame {t} - region_loss: {region_loss}")

            attn_com = get_com(attn_topk)
            attn_com_t1 = get_com(attn_t1_topk)
            mask_com = get_com(mask)
            mask_com_t1 = get_com(mask_t1)

            speed_loss = torch.norm((attn_com_t1 - attn_com) - (mask_com_t1 - mask_com), p=2)
            print(f"[2] obj {obj_idx} - frame {t} -> {t+1} - speed_loss: {speed_loss}")

            layout_flow = box_center(box1) - box_center(box0)
            flow_map_layout = torch.zeros((H, W, 2), device=device)
            flow_mask = mask.bool()
            flow_map_layout[flow_mask] = layout_flow

            amf_idx = t * (latent_frames + 1) + 1
            pred_flow_map = amf_map[amf_idx].view(H, W, 2)
            attn_mask = attn / (attn.max() + 1e-8)
            attn_soft = attn_mask.unsqueeze(-1)
            pred_flow_map = pred_flow_map * attn_soft
            # flow_loss = F.mse_loss(pred_flow_map[flow_mask],
                                # flow_map_layout[flow_mask],
                                # reduction="mean")
            # flow_loss = F.mse_loss(pred_flow_map, flow_map_layout, reduction="mean")

            mse_per_pixel = F.mse_loss(pred_flow_map, flow_map_layout, reduction="none")  # shape: (H, W, 2)
            flow_loss = (mse_per_pixel * attn_soft).sum() / attn_soft.sum()

            print(f"[3] obj {obj_idx} - frame {t} -> {t+1} - flow_loss: {flow_loss}")
            obj_loss += (flow_loss + speed_loss * 5 + region_loss)
            # obj_loss += (flow_loss + speed_loss + region_loss) * (2 * latent_frames - t) / latent_frames
            # obj_loss += (speed_loss + region_loss) * (2 * latent_frames - t) / latent_frames
            # obj_loss += (speed_loss + region_loss)
            del mask, attn, attn_soft, flow_map_layout, pred_flow_map
            # del mask, attn

        print(f"[4] obj {obj_idx} - max_frame_info_entropy_loss: {max_frame_info_entropy_loss}")
        max_obj_info_entropy_loss = torch.max(max_obj_info_entropy_loss, max_frame_info_entropy_loss)
        obj_loss = obj_loss / object_number
        print(f"[4] obj {obj_idx} - obj_loss: {obj_loss}")
        loss += obj_loss
        del token_map, obj_loss
    # global_flow_loss = compute_global_flow_loss()
    # loss = (loss + global_flow_loss) * 50.0
    print(f"[5] max_obj_info_entropy_loss: {max_obj_info_entropy_loss}")
    print(f"[5] region&motion loss: {loss}")
    loss = (loss + max_obj_info_entropy_loss)
    return loss