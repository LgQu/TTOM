#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
from pathlib import Path

# ===========================
# 固定路径配置
# ===========================
PROJECT_ROOT      = Path("/Path/To/Your/Project")
ATTENTION_PTS_DIR = PROJECT_ROOT / "data/attn_maps/wan2.1-t2v-14b"
DINO_RESULTS_ROOT = PROJECT_ROOT / "data/dino_results_batch"
SUMMARY_DIR       = PROJECT_ROOT / "data/miou_summary"
CURVE_DPI         = 200

# 评测与可视化参数
HEAT_H, HEAT_W     = 30, 52          # 注意力图网格大小
VIS_DPI            = 200
MASK_CONTOUR_LEVEL = 0.5             # 轮廓阈值
HEATMAP_CMAP       = "viridis"
NORM_MODE_FOR_IOU = "minmax"  # 可选: "minmax" / "percentile"
NORM_MODE_FOR_JSD = "softmax"     # JSD 推荐 softmax，变成空间分布
# ============== 工具函数 ==============

# ===== 新增：归一化与指标 =====
def normalize_spatial(arr: np.ndarray, mode: str = "minmax", eps: float = 1e-8) -> np.ndarray:
    """
    对 2D 数组做空间归一化：
      - 'minmax': 标准 minmax
      - 'percentile': 先按 [p1, p99] 裁剪再 minmax
      - 'softmax': 视为未归一化对数分数，对空间做 softmax（适合做分布度量）
    """
    a = arr.astype(np.float32)
    if mode == "minmax":
        vmin, vmax = float(a.min()), float(a.max())
        if vmax - vmin < eps:
            return np.zeros_like(a)
        return (a - vmin) / (vmax - vmin)
    elif mode == "percentile":
        p1, p99 = np.percentile(a, 1), np.percentile(a, 99)
        a = np.clip(a, p1, p99)
        vmin, vmax = float(a.min()), float(a.max())
        if vmax - vmin < eps:
            return np.zeros_like(a)
        return (a - vmin) / (vmax - vmin)
    elif mode == "softmax":
        x = a - a.max()
        expx = np.exp(x, dtype=np.float64)
        s = expx.sum()
        if s < eps:
            return np.full_like(a, 1.0 / a.size)
        return (expx / s).astype(np.float32)
    else:
        raise ValueError(f"Unknown norm mode: {mode}")

def dice_soft(A: np.ndarray, M: np.ndarray, eps: float = 1e-8) -> float:
    A = np.clip(A, 0.0, 1.0); M = np.clip(M, 0.0, 1.0)
    inter = float(np.sum(A * M))
    return float((2.0 * inter) / (A.sum() + M.sum() + eps))

def jsd_2d(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12, log_base: float = 2.0) -> float:
    """
    对 2D map 的 Jensen–Shannon Divergence。
    先把 P, Q 归一化为空间分布（和 softmax/percentile/minmax之一配合）。
    """
    P = P.astype(np.float64); Q = Q.astype(np.float64)
    P = np.clip(P, 0.0, None); Q = np.clip(Q, 0.0, None)
    P_sum, Q_sum = P.sum(), Q.sum()
    if P_sum <= eps: P = np.full_like(P, 1.0 / P.size)
    else:            P = P / P_sum
    if Q_sum <= eps: Q = np.full_like(Q, 1.0 / Q.size)
    else:            Q = Q / Q_sum
    M = 0.5 * (P + Q)
    # KL(P||M) + KL(Q||M)
    kl_PM = np.sum(np.where(P > 0, P * (np.log(P + eps) - np.log(M + eps)), 0.0))
    kl_QM = np.sum(np.where(Q > 0, Q * (np.log(Q + eps) - np.log(M + eps)), 0.0))
    jsd = 0.5 * (kl_PM + kl_QM)
    if log_base == 2.0:
        jsd = jsd / np.log(2.0)
    return float(jsd)


def sampled_video_indices(N: int, T: int) -> np.ndarray:
    """把视频 N 帧均匀采样/映射到 T 帧（包含首尾）"""
    assert N >= 1 and T >= 1
    return np.round(np.linspace(0, N - 1, T)).astype(int)

def load_attn_all_layers(
    pt_path: Path,
    step_id: Optional[int] = None,
    inst_id: Optional[int] = None,
) -> Tuple[List[int], Dict[int, np.ndarray], int]:
    """
    从 .pt 里取指定 step/inst 的所有 layer 的 attn_map（形状 [T,30,52]）
    返回: (layers, {layer_id: [T,30,52]}, T)
    """
    data = torch.load(str(pt_path), map_location="cpu")
    steps = sorted(list(data["timestep"].keys()))
    if not steps:
        raise ValueError("No 'timestep' found in pt.")

    if step_id is None or step_id not in steps:
        if step_id is None:
            step_id = steps[0]
        else:
            step_id = min(steps, key=lambda s: abs(s - step_id))
        print(f"[INFO] use step_id={step_id}")

    layers = sorted(list(data["timestep"][step_id]["layers"].keys()))
    if not layers:
        raise ValueError(f"No layers found at step {step_id}.")

    # 选择 inst
    first_layer = layers[0]
    insts = sorted(list(data["timestep"][step_id]["layers"][first_layer]["insts"].keys()))
    if not insts:
        raise ValueError(f"No instances found at step {step_id}, layer {first_layer}.")
    if inst_id is None or inst_id not in insts:
        raise ValueError(f"inst_id={inst_id} not found in PT. Available insts: {insts}")

    attn_by_layer: Dict[int, np.ndarray] = {}
    T = None
    for lid in layers:
        attn = data["timestep"][step_id]["layers"][lid]["insts"][inst_id]["attn_map"]  # [T,30,52]
        arr = attn.detach().cpu().numpy().astype(np.float32)
        if T is None:
            T = arr.shape[0]
        attn_by_layer[lid] = arr
    return layers, attn_by_layer, T

def find_video_dir_by_pid(pid: int) -> Path:
    """在 DINO_RESULTS_ROOT 里找形如 pid{pid}_[...] 的目录"""
    candidates = sorted([p for p in DINO_RESULTS_ROOT.iterdir() if p.is_dir() and p.name.startswith(f"pid{pid}_")])
    if not candidates:
        raise FileNotFoundError(f"No video directory found for pid{pid}_* under {DINO_RESULTS_ROOT}")
    if len(candidates) > 1:
        print(f"[WARN] multiple candidates found -> choose first: {[c.name for c in candidates]}")
    return candidates[0]

def find_pt_by_pid(pid: int) -> Path:
    """在 ATTENTION_PTS_DIR 里找以 pid{pid}_ 开头的 .pt 文件"""
    candidates = sorted(ATTENTION_PTS_DIR.glob(f"pid{pid}_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No PT file like 'pid{pid}_*.pt' under {ATTENTION_PTS_DIR}")
    if len(candidates) > 1:
        print(f"[WARN] multiple PT candidates -> choose first: {[c.name for c in candidates]}")
    return candidates[0]

def parse_inst_dirs(masks_root: Path) -> List[Tuple[int, Path]]:
    """
    返回 [(inst_id, inst_dir)], inst_dir 形如 '0_[red fox]'
    """
    out = []
    if not masks_root.exists():
        raise FileNotFoundError(f"Missing masks dir: {masks_root}")
    for d in sorted(masks_root.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"^(\d+)_", d.name)
        if m:
            out.append((int(m.group(1)), d))
    if not out:
        raise FileNotFoundError(f"No inst directories like '0_[...]' under {masks_root}")
    return out

def infer_N_from_all_masks(masks_root: Path, inst_dirs: List[Tuple[int, Path]]) -> int:
    """
    仅从 mask 文件推断视频总帧数 N：取所有实例目录里 frame_XXXXXX.png 的最大编号 + 1
    """
    max_idx = -1
    pattern = re.compile(r"^frame_(\d{6})\.png$")
    for _, inst_dir in inst_dirs:
        for f in inst_dir.iterdir():
            if not f.is_file():
                continue
            m = pattern.match(f.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
    if max_idx < 0:
        raise FileNotFoundError(f"No mask frames found under {masks_root}")
    N = max_idx + 1
    return N

def read_mask_png(path: Path) -> np.ndarray:
    """读取单帧 mask PNG -> [H,W] float32 in [0,1]"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    arr = img.astype(np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr

def mask_to_30x52(mask: np.ndarray) -> np.ndarray:
    """任意尺寸 mask -> [30,52]，用 INTER_AREA 下采样"""
    out = cv2.resize(mask, (HEAT_W, HEAT_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = np.clip(out, 0.0, 1.0)
    return out

def minmax_norm(frame: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vmin, vmax = float(frame.min()), float(frame.max())
    if vmax - vmin < eps:
        return np.zeros_like(frame, dtype=np.float32)
    return (frame - vmin) / (vmax - vmin)

def soft_iou(A: np.ndarray, M: np.ndarray, eps: float = 1e-8) -> float:
    A = np.clip(A, 0.0, 1.0); M = np.clip(M, 0.0, 1.0)
    inter = float(np.sum(A * M))
    union = float(np.sum(A + M - A * M))
    return float(inter / (union + eps))

def collect_masks_for_T(inst_dir: Path, N: int, T: int) -> np.ndarray:
    """
    根据 N->T 映射，收集 inst_dir 下 T 帧 mask，统一到 [T,30,52]
    缺帧则置零并告警一次
    """
    vid_indices = sampled_video_indices(N, T)
    masks = []
    missed = 0
    for vi in vid_indices:
        fn = inst_dir / f"frame_{int(vi):06d}.png"
        if fn.exists():
            m = mask_to_30x52(read_mask_png(fn))
        else:
            missed += 1
            m = np.zeros((HEAT_H, HEAT_W), dtype=np.float32)
        masks.append(m)
    if missed > 0:
        print(f"[WARN] {inst_dir.name}: missing {missed}/{T} masks; filled with zeros.")
    return np.stack(masks, axis=0)  # [T,30,52]

def visualize_layer_firstframe(
    attn: np.ndarray,         # [T,30,52] 原始注意力
    masks: np.ndarray,        # [T,30,52] 软 mask
    out_png: Path,
    layer_id: int,
    vid_indices: np.ndarray,
    title_prefix: str = "",
    dpi: int = VIS_DPI,
):
    """
    仅第一帧可视化：
    - 背景：注意力热图(归一化)
    - 叠加：mask 的等值线(0.5) 轮廓（无透明度）
    - 无坐标刻度、无网格、无边框
    """
    t = 0  # 只画第一帧
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.6), constrained_layout=True)

    A_norm = minmax_norm(attn[t])
    M = np.clip(masks[t], 0.0, 1.0)

    # 注意力热图（不透明）
    ax.imshow(A_norm, cmap=HEATMAP_CMAP, origin="upper",
              vmin=0.0, vmax=1.0, extent=(0, HEAT_W, HEAT_H, 0), aspect="equal")

    # 叠加 mask 的等值线（无透明度）
    try:
        ax.contour(M, levels=[MASK_CONTOUR_LEVEL], colors="red", linewidths=2.0)
    except Exception:
        pass

    # 标题 & 关闭坐标/边框
    ax.set_title(f"Layer {layer_id} | t=0 (vid {int(vid_indices[t])})", fontsize=11)
    ax.axis("off")  # 无刻度、无边框

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12, y=1.02)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ============== 主流程（只需要 pid） ==============

def run_for_pid(pid: int, step_id: Optional[int] = None, only_inst: Optional[int] = None):
    # 1) 定位视频目录 + masks 根
    video_dir = find_video_dir_by_pid(pid)
    masks_root = video_dir / "masks"
    inst_dirs = parse_inst_dirs(masks_root)
    if only_inst is not None:
        inst_dirs = [x for x in inst_dirs if x[0] == only_inst]
        if not inst_dirs:
            raise FileNotFoundError(f"inst={only_inst} not found under {masks_root}")

    # 2) 从所有 mask 直接推断视频帧数 N
    N = infer_N_from_all_masks(masks_root, inst_dirs)
    print(f"[INFO] pid{pid} -> video_dir={video_dir.name}, N(inferred from masks)={N}")

    # 3) 找到 .pt（供所有 inst 共用）
    pt_path = find_pt_by_pid(pid)
    print(f"[INFO] PT file: {pt_path.name}")

    # 4) 结果目录：video_dir/miou
    miou_root = video_dir / "miou"
    miou_root.mkdir(parents=True, exist_ok=True)

    # 5) 汇总
    summary_rows = []
    summary_json = {
        "pid": pid,
        "video_dir": str(video_dir),
        "pt_path": str(pt_path),
        "N_inferred": N,
        "insts": {}
    }

    # 针对每个 inst 做一次评测
    for inst_id, inst_dir in inst_dirs:
        print(f"\n[INFO] Evaluating inst={inst_id} @ {inst_dir.name}")

        # 读取 PT（指定相同 inst_id）
        layers, attn_by_layer, T = load_attn_all_layers(pt_path, step_id=step_id, inst_id=inst_id)
        vid_indices = sampled_video_indices(N, T)

        # 读取 T 帧 masks -> [T,30,52]
        masks_T = collect_masks_for_T(inst_dir, N=N, T=T)

        # 逐层度量
        per_layer_means = []   # 仍以 IoU 作为 overall 的依据（与原版一致）
        rows = []
        inst_out_dir = miou_root / f"inst{inst_id}"
        inst_out_dir.mkdir(parents=True, exist_ok=True)

        # 可视化目录
        vis_dir = inst_out_dir / "overlays"
        vis_dir.mkdir(parents=True, exist_ok=True)

        inst_result = {
            "layers": {},
            "overall_mean_soft_iou": 0.0,
            "T": T, "N": N,
            "norm_for_iou": NORM_MODE_FOR_IOU,
            "norm_for_jsd": NORM_MODE_FOR_JSD,
        }

        for lid, attn in attn_by_layer.items():
            iou_vals, dice_vals, jsd_vals = [], [], []

            for t in range(T):
                # IoU/Dice：更鲁棒的 percentile 归一化（或改成 'minmax'）
                A_iou = normalize_spatial(attn[t], mode=NORM_MODE_FOR_IOU)
                # JSD：softmax 转为空间分布
                A_jsd = normalize_spatial(attn[t], mode=NORM_MODE_FOR_JSD)
                M = np.clip(masks_T[t], 0.0, 1.0)

                iou_vals.append(soft_iou(A_iou, M))
                dice_vals.append(dice_soft(A_iou, M))
                jsd_vals.append(jsd_2d(A_jsd, M))

            mean_iou  = float(np.mean(iou_vals))
            mean_dice = float(np.mean(dice_vals))
            mean_jsd  = float(np.mean(jsd_vals))

            per_layer_means.append(mean_iou)  # overall 仍按 IoU 聚合

            inst_result["layers"][str(lid)] = {
                "mean_soft_iou":  mean_iou,
                "mean_dice":      mean_dice,
                "mean_jsd":       mean_jsd,
                "per_frame_soft_iou": iou_vals,
                "per_frame_dice":     dice_vals,
                "per_frame_jsd":      jsd_vals,
            }

            # 写行
            for t in range(T):
                rows.append({
                    "pid": pid,
                    "inst": inst_id,
                    "layer": lid,
                    "attn_frame": t,
                    "video_frame": int(vid_indices[t]),
                    "soft_iou": float(iou_vals[t]),
                    "dice":     float(dice_vals[t]),
                    "jsd":      float(jsd_vals[t]),
                })

            # 仅第一帧可视化（无透明度、无刻度/边框）；标题包含三指标
            out_png = vis_dir / f"layer_{lid}_t0.png"
            visualize_layer_firstframe(
                attn, masks_T, out_png, layer_id=lid, vid_indices=vid_indices,
                title_prefix=f"pid{pid} inst{inst_id} | IoU={mean_iou:.3f}  Dice={mean_dice:.3f}  JSD={mean_jsd:.3f}"
            )

        # 总均值（保持与你原来一致：基于 per-layer IoU 均值）
        overall_mean = float(np.mean(per_layer_means)) if per_layer_means else 0.0
        inst_result["overall_mean_soft_iou"] = overall_mean

        # 保存 CSV/JSON
        import csv
        csv_path = inst_out_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["pid", "inst", "layer", "attn_frame", "video_frame", "soft_iou", "dice", "jsd"]
            )
            writer.writeheader()
            writer.writerows(rows)

        json_path_out = inst_out_dir / "results.json"
        with open(json_path_out, "w", encoding="utf-8") as f:
            json.dump(inst_result, f, ensure_ascii=False, indent=2)

        print(f"[SAVE] {csv_path}")
        print(f"[SAVE] {json_path_out}")
        print(f"[SAVE] overlays -> {vis_dir}")

        # 汇总
        summary_json["insts"][str(inst_id)] = {
            "overall_mean_soft_iou": overall_mean,
            "num_layers": len(attn_by_layer),
            "T": T,
            "N": N,
            "inst_dir": str(inst_dir),
            "results_dir": str(inst_out_dir),
        }
        summary_rows.append({
            "pid": pid,
            "inst": inst_id,
            "overall_mean_soft_iou": overall_mean,
            "num_layers": len(attn_by_layer),
            "T": T,
            "N": N,
        })


    # 汇总文件
    summary_json_path = miou_root / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    import csv
    summary_csv_path = miou_root / "summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pid", "inst", "overall_mean_soft_iou", "num_layers", "T", "N"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[DONE] pid{pid}")
    print(f"  Summary JSON: {summary_json_path}")
    print(f"  Summary CSV : {summary_csv_path}")

def find_all_pids_in_pt_root() -> List[int]:
    """
    从 ATTENTION_PTS_DIR 下扫描 'pid*.pt' 解析出所有 pid（去重、升序）。
    """
    pids = set()
    pat = re.compile(r"pid(\d+)_")
    for p in sorted(ATTENTION_PTS_DIR.glob("pid*.pt")):
        m = pat.search(p.name)
        if m:
            pids.add(int(m.group(1)))
    return sorted(pids)

def run_for_all_pids(
    step_id: Optional[int] = None,
    only_inst: Optional[int] = None,
    skip_done: bool = False,
    pid_list: Optional[List[int]] = None,
):
    """
    批量执行 run_for_pid。
    - 默认从 ATTENTION_PTS_DIR 扫描 pid；也可通过 pid_list 手动传入。
    - skip_done=True 时，如果对应视频目录下已有 miou/summary.json 就跳过。
    """
    if pid_list is None:
        pid_list = find_all_pids_in_pt_root()
    if not pid_list:
        print(f"[WARN] No pid found under {ATTENTION_PTS_DIR}")
        return

    print(f"[INFO] Will process {len(pid_list)} pid(s): {pid_list}")
    for pid in pid_list:
        if skip_done:
            try:
                vid_dir = find_video_dir_by_pid(pid)
                if (vid_dir / "miou" / "summary.json").exists():
                    print(f"[SKIP DONE] pid{pid}: {vid_dir.name}/miou/summary.json exists.")
                    continue
            except Exception as e:
                # 没有匹配视频目录则继续尝试跑，run_for_pid 内部会给出清晰报错
                pass

        try:
            run_for_pid(pid=pid, step_id=step_id, only_inst=only_inst)
        except Exception as e:
            print(f"[ERROR] pid{pid}: {e}")

def global_curve_from_instance_jsons(
    dino_root: Path = DINO_RESULTS_ROOT,
    out_dir: Path = SUMMARY_DIR,
    curve_name_all: str = "layer_curve_all_metrics_CI.png",
    dpi: int = CURVE_DPI,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """
    聚合 mean_soft_iou / mean_dice / mean_jsd，并画含95%置信区间的曲线图。
    输出:
      - CSV: layer_stats_from_instances.csv
      - JSON: layer_stats_from_instances.json （数值均为 Python 标量；NaN/Inf 已置 None）
      - PNG: layer_curve_all_metrics_CI.png（三条指标+CI）
      - PNG: layer_curve_iou_CI.png / layer_curve_dice_CI.png / layer_curve_jsd_CI.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    inst_jsons = sorted(dino_root.rglob("miou/inst*/results.json"))
    if not inst_jsons:
        print(f"[WARN] No instance results.json found under {dino_root}")
        return None, None, None, None

    # --- helpers ---
    import math
    def to_py_number(x):
        # 统一把 numpy 标量 / Python 标量转为内置 float/int；非有限数->None
        if x is None:
            return None
        if isinstance(x, (np.generic,)):  # np.float32, np.int64 等
            x = x.item()
        if isinstance(x, (int,)):
            return int(x)
        if isinstance(x, (float,)):
            return None if (math.isnan(x) or math.isinf(x)) else float(x)
        # 其它类型不应出现，返回 None 以防止 json.dump 报错
        return None

    def safe_list_to_float(arr):
        # 用于绘图：把 None -> np.nan
        return np.array([ (float(v) if (v is not None) else np.nan) for v in arr ], dtype=float)

    agg = {"iou": {}, "dice": {}, "jsd": {}}
    used = 0
    skipped = 0

    for jp in inst_jsons:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                jd = json.load(f)
        except Exception as e:
            print(f"[SKIP] bad json {jp}: {e}")
            skipped += 1
            continue

        layers_dict = jd.get("layers", {})
        if not isinstance(layers_dict, dict) or not layers_dict:
            skipped += 1
            continue

        for k, v in layers_dict.items():
            try:
                lid = int(k)
            except Exception:
                continue
            for metric, key in [("iou", "mean_soft_iou"), ("dice", "mean_dice"), ("jsd", "mean_jsd")]:
                val = v.get(key, None)
                if val is not None:
                    try:
                        val = float(val)
                        agg[metric].setdefault(lid, []).append(val)
                    except Exception:
                        pass
        used += 1

    layer_set = set(agg["iou"].keys()) | set(agg["dice"].keys()) | set(agg["jsd"].keys())
    if not layer_set:
        print(f"[WARN] No layer metric data aggregated from {len(inst_jsons)} instance jsons.")
        return None, None, None, None
    layers_sorted = sorted(layer_set)

    def _stat(values: List[float]):
        vals = np.array(values, dtype=np.float64)
        n = int(vals.size)
        if n == 0:
            return 0, None, None, None
        mean = float(vals.mean())
        std  = float(vals.std(ddof=0))
        ci   = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
        return n, mean, std, ci

    rows: List[Dict[str, object]] = []
    for lid in layers_sorted:
        n_iou, m_iou, s_iou, c_iou = _stat(agg["iou"].get(lid, []))
        n_dice, m_dice, s_dice, c_dice = _stat(agg["dice"].get(lid, []))
        n_jsd, m_jsd, s_jsd, c_jsd = _stat(agg["jsd"].get(lid, []))

        rows.append(dict(
            layer=int(lid),
            iou_count=int(n_iou),  iou_mean=to_py_number(m_iou), iou_std=to_py_number(s_iou), iou_ci=to_py_number(c_iou),
            dice_count=int(n_dice), dice_mean=to_py_number(m_dice), dice_std=to_py_number(s_dice), dice_ci=to_py_number(c_dice),
            jsd_count=int(n_jsd),  jsd_mean=to_py_number(m_jsd), jsd_std=to_py_number(s_jsd), jsd_ci=to_py_number(c_jsd),
        ))

    # --- 保存 CSV ---
    import csv
    csv_path = out_dir / "layer_stats_from_instances.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- 保存 JSON（保证全是 Python 标量；NaN/Inf->None） ---
    json_path = out_dir / "layer_stats_from_instances.json"
    payload = {"layers": rows, "used": int(used), "skipped": int(skipped)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    # --- 画曲线（含 CI），对 None 用 np.nan 绘制 ---
    x = [r["layer"] for r in rows]

    def plot_with_ci(x, y_mean, y_ci, label):
        ym = safe_list_to_float(y_mean)
        yc = safe_list_to_float(y_ci)
        plt.plot(x, ym, label=label, linewidth=2)
        plt.fill_between(x, ym - yc, ym + yc, alpha=0.2)

    # 合并图（三条曲线一张）
    plt.figure(figsize=(8, 4), dpi=dpi)
    plot_with_ci(x, [r["iou_mean"]  for r in rows], [r["iou_ci"]  for r in rows], "IoU")
    plot_with_ci(x, [r["dice_mean"] for r in rows], [r["dice_ci"] for r in rows], "Dice")
    plot_with_ci(x, [r["jsd_mean"]  for r in rows], [r["jsd_ci"]  for r in rows], "JSD")
    plt.xlabel("Layer ID"); plt.ylabel("Metric"); plt.grid(True, alpha=0.3); plt.legend()
    curve_all = out_dir / curve_name_all
    plt.tight_layout(); plt.savefig(curve_all); plt.close()

    # 单独三张
    def _plot_one(metric: str, ylabel: str) -> Path:
        ym = [r[f"{metric}_mean"] for r in rows]
        yc = [r[f"{metric}_ci"]   for r in rows]
        plt.figure(figsize=(8, 4), dpi=dpi)
        plot_with_ci(x, ym, yc, ylabel)
        plt.xlabel("Layer ID"); plt.ylabel(ylabel); plt.grid(True, alpha=0.3)
        outp = out_dir / f"layer_curve_{metric}_CI.png"
        plt.tight_layout(); plt.savefig(outp); plt.close()
        return outp

    curve_iou  = _plot_one("iou",  "IoU")
    curve_dice = _plot_one("dice", "Dice")
    curve_jsd  = _plot_one("jsd",  "JSD")

    print(f"[GLOBAL] used {used}, skipped {skipped}")
    print(f"[GLOBAL] CSV : {csv_path}")
    print(f"[GLOBAL] JSON: {json_path}")
    print(f"[GLOBAL] Curves with CI: {curve_all}, {curve_iou}, {curve_dice}, {curve_jsd}")
    return curve_all, csv_path, json_path, curve_jsd




# ============== CLI ==============

def main():
    parser = argparse.ArgumentParser(
        description="Compute soft mIoU (attn vs masks) by pid; visualize only first frame (no alpha, no ticks)."
    )
    parser.add_argument("--pid", type=int, default=None, help="Single video PID (e.g., 0)")
    parser.add_argument("--step", type=int, default=40, help="Step ID to use (default: nearest/first in PT)")
    parser.add_argument("--inst", type=int, default=None, help="Only evaluate a specific inst id; default: all found")

    # 新增批量参数
    parser.add_argument("--run_all", action="store_true", help="Process ALL pids found under ATTENTION_PTS_DIR")
    parser.add_argument("--pids", type=str, default=None, help="Comma/space separated pid list, e.g. '0,3,7'")
    parser.add_argument("--skip_done", action="store_true", help="Skip pid whose miou/summary.json already exists")
    parser.add_argument("--plot_global", action="store_true", help="Plot global curve")

    args = parser.parse_args()

    if args.plot_global:
        global_curve_from_instance_jsons()
        return

    if args.run_all or args.pids:
        if args.pids:
            pid_list = [int(x) for x in re.split(r"[,\s]+", args.pids.strip()) if x]
        else:
            pid_list = None  # 自动从 ATTENTION_PTS_DIR 扫描
        run_for_all_pids(
            step_id=args.step,
            only_inst=args.inst,
            skip_done=args.skip_done,
            pid_list=pid_list,
        )
        global_curve_from_instance_jsons()
        return

    # 单个 pid 模式
    if args.pid is None:
        parser.error("Please specify --pid for single run, or use --run_all / --pids for batch.")
    run_for_pid(pid=args.pid, step_id=args.step, only_inst=args.inst)


if __name__ == "__main__":
    main()
