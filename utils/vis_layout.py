# render_layouts.py
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 你项目里的工具模块，需在 PYTHONPATH 下可导入
import utils

# 画布与输出配置
CANVAS_W, CANVAS_H = 832, 480          # 你的场景分辨率
NUM_FRAMES = 21                         # 插值目标帧数
OUTPUT_ROOT = Path("/scratch/e1351271/video_gen/cache/layout")

# 预设 20 个高对比 RGB 颜色（0~255）
PRESET20 = [
    (31,119,180),(214,39,40),(44,160,44),(148,103,189),(255,127,14),
    (140,86,75),(23,190,207),(57,59,121),(0,98,155),(0,153,51),
    (153,51,153),(204,102,0),(50,50,50),(166,118,29),(117,112,179),
    (231,41,138),(188,189,34),(127,127,127),(0,128,128),(227,119,194),
]

def vis_layout(
    layout_list,
    output_path="layout_outline.mp4",
    fps=4,
    obj_thickness=5,     # 物体边框向外扩的像素粗细
    canvas_border=5,     # 画布外框粗细（黑色）
    palette="preset20",  # 'preset20' 或自定义 [(R,G,B), ...]
):
    """
    仅以边框可视化（白底 + 画布黑边 + 物体粗边，且边框只向外拓宽）。
    layout_list: List[{'masks': List[Tensor(H,W) 或 np.ndarray]}]
    """
    import imageio
    import torch
    import numpy as np
    import torch.nn.functional as F

    assert len(layout_list) > 0, "layout_list 不能为空"
    T = len(layout_list[0]["masks"])
    H, W = layout_list[0]["masks"][0].shape
    N = len(layout_list)

    # 颜色表
    if palette == "preset20":
        colors = PRESET20
    elif isinstance(palette, (list, tuple)) and len(palette) > 0:
        colors = list(palette)
    else:
        colors = PRESET20

    # 工具函数们
    def _to_bin(x):
        # -> torch.float (1,1,H,W), 值为 0/1
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = (x.float() > 0).float().unsqueeze(0).unsqueeze(0)
        return x

    def _dilate(x, iters=1):
        # 3x3 结构元，迭代膨胀；每次约向外扩 1px
        k = max(1, int(iters))
        for _ in range(k):
            x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x

    def _outer_band_only(x, thickness):
        """
        只取 '外侧边框'：
        outer = dilate(mask, thickness) AND (NOT mask)
        """
        d = _dilate(x, thickness)           # (1,1,H,W)
        outer = (d > 0.5) & (x <= 0.5)      # bool
        return outer.squeeze().cpu().numpy().astype(bool)  # (H,W)

    # 绘制
    frames = []
    for t in range(T):
        # 1) 白底
        frame = np.full((H, W, 3), 255, dtype=np.uint8)

        # 2) 逐实例：向外拓宽的边框
        for idx, inst in enumerate(layout_list):
            m = inst["masks"][t]
            edge = _outer_band_only(_to_bin(m), obj_thickness)  # (H,W) bool
            if edge.any():
                c = colors[idx % len(colors)]
                frame[edge] = c  # 覆盖绘制

        # 3) 最后加“画布黑边框”（覆盖一切）
        b = max(1, int(canvas_border))
        frame[:b, :, :] = 0
        frame[-b:, :, :] = 0
        frame[:, :b, :] = 0
        frame[:, -b:, :] = 0

        frames.append(frame)

    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")
    print(f"Video saved to {output_path} | size={W}x{H} | outward_thickness={obj_thickness} | canvas_border={canvas_border} | colors={len(colors)}")




def _to_box(obj: Dict[str, Any]) -> List[int]:
    """
    统一把对象里的位置信息转成 box=[x,y,w,h]
    支持:
      - {"box":[x,y,w,h]}
      - {"x":..,"y":..,"w":..,"h":..}
    """
    if "box" in obj and isinstance(obj["box"], list) and len(obj["box"]) == 4:
        x, y, w, h = obj["box"]
    else:
        x = obj.get("x", 0)
        y = obj.get("y", 0)
        w = obj.get("w", 100)
        h = obj.get("h", 100)
    return [int(x), int(y), int(w), int(h)]

def _clamp_box(box: List[int], wmax: int, hmax: int) -> List[int]:
    x, y, w, h = box
    # 简单夹紧，确保不出界&最小尺寸
    w = max(10, min(w, wmax - 10))
    h = max(10, min(h, hmax - 10))
    x = max(0, min(x, wmax - w))
    y = max(0, min(y, hmax - h))
    return [x, y, w, h]

def _extract_frames_mapping(entry_layout: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    兼容两种 layout 结构：
      - 直接是 {"1":[...],"2":[...]}
      - 或 {"reasoning":"...","frames":{"1":[...],...}}
    返回统一的 frames 映射
    """
    if "frames" in entry_layout and isinstance(entry_layout["frames"], dict):
        return entry_layout["frames"]
    return entry_layout  # 已经是 frames 映射

def render_from_json(json_path: str,
                     canvas_w: int = CANVAS_W,
                     canvas_h: int = CANVAS_H,
                     num_frames: int = NUM_FRAMES,
                     output_root: Path = OUTPUT_ROOT):
    """
    读取含多条 entry 的 JSON，逐条渲染 layout 可视化视频。
    输出：/scratch/e1351271/video_gen/cache/layout/{json文件名}/pid{entry_id}.mp4
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {json_path}")

    # 目标输出目录：layout/{json文件名}/
    out_dir = output_root / json_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 遍历每一条 entry
    keys = sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    print(f"📂 Input:  {json_path}")
    print(f"📄 Output: {out_dir} (one mp4 per entry)\n")

    for k in keys:
        entry = data[k]
        prompt = entry.get("prompt", "")
        layout_block = entry.get("layout", None)
        if not layout_block:
            print(f"⚠️  Entry {k}: no 'layout' found, skip.")
            continue

        frames_map = _extract_frames_mapping(layout_block)
        if not isinstance(frames_map, dict) or not frames_map:
            print(f"⚠️  Entry {k}: invalid 'frames' mapping, skip.")
            continue

        # ===== 构建 instance_boxes: inst_id -> {frame_idx: [x,y,w,h]} =====
        # 同时记录 id->name
        from collections import defaultdict
        instance_boxes = defaultdict(dict)   # type: Dict[int, Dict[int, List[int]]]
        instance_names: Dict[int, str] = {}

        # 将 "1".."6" 转成 0-based 索引；并统一 box；夹紧到画布
        for frame_str, obj_list in frames_map.items():
            try:
                fidx = int(frame_str) - 1  # "1" -> 0
            except ValueError:
                print(f"   ⚠️  Entry {k}: frame key '{frame_str}' not int-like, skip frame.")
                continue

            if not isinstance(obj_list, list):
                print(f"   ⚠️  Entry {k}: frame '{frame_str}' objects not a list, skip frame.")
                continue

            for obj in obj_list:
                if not isinstance(obj, dict):
                    continue
                inst_id = int(obj.get("id", 0))
                name = str(obj.get("name", ""))
                box = _to_box(obj)
                box = _clamp_box(box, canvas_w, canvas_h)

                instance_boxes[inst_id][fidx] = box
                instance_names[inst_id] = name

        if not instance_boxes:
            print(f"⚠️  Entry {k}: empty instance_boxes, skip.")
            continue

        # ===== 调用插值：输出 (inst_id -> masks[frame]) =====
        # interpolate_instance_masks(instance_boxes, H, W, num_frames)
        # 期望返回形如 {inst_id: np.ndarray(shape=(num_frames,H,W), dtype=bool/0-1)}
        masks_dict = utils.interpolate_instance_masks(
            instance_boxes,
            canvas_h,
            canvas_w,
            num_frames=num_frames
        )

        # ===== 组织 layout_list，供 vis_layout 使用 =====
        layout_list = []
        for inst_id in sorted(masks_dict.keys()):
            masks = masks_dict[inst_id]  # (num_frames, H, W)
            layout_list.append({
                "id": inst_id,
                "name": instance_names.get(inst_id, ""),
                "masks": [masks[i] for i in range(masks.shape[0])]
            })

        # ===== 保存可视化视频 =====
        out_path = out_dir / f"pid{k}.mp4"
        print(f"➡️  Entry {k}: '{prompt}'")
        print(f"    Instances: {len(layout_list)}; Saving to: {out_path}")
        vis_layout(layout_list, output_path=str(out_path))

    print("\n🎉 Done.")

if __name__ == "__main__":
    # 使用示例：
    # python render_layouts.py /scratch/e1351271/video_gen/cache/prompts/1_consistent_attr.gpt-4o.layout.json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/scratch/e1351271/video_gen/cache/7_numeracy.gpt-4o.json")
    args = parser.parse_args()
    render_from_json(args.json_path)
