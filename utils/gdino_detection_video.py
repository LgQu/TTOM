import os
import re
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv

# --------- Config (edit these) ----------
VIDEO_DIR    = "/scratch/e1351271/video_gen/ttt-lm/data/t2v_compbench/cache_train_motion_gpt-4o/wan_full_enriched_wo"
OUTPUT_ROOT  = "data/dino_results_batch"
BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25

CONFIG_PATH  = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
# ----------------------------------------

# ==== SAM2 配置（逐帧分割） ====
USE_SAM2                  = True            # 关掉则不做分割
SAM2_CFG                  = "configs/sam2.1/sam2.1_hiera_l.yaml"     # 按你的权重选择
SAM2_CKPT                 = "sam2/checkpoints/sam2.1_hiera_large.pt"           # 修改为你的权重路径
SEGMENT_ONLY_FIRST_FRAME  = False           # True 仅第1帧做分割（最快）
SEGMENT_EVERY_N           = 1               # 每 N 帧分割一次（1=每帧）
SAVE_MASKS                = True            # 保存单帧掩码 PNG
MASKS_SUBDIR              = "masks"         # 掩码保存目录（视频输出子目录下）
MASK_ALPHA_FILL           = 0.35            # 第1帧可视化时，mask 叠加透明度
# =========================================

# sanity checks
assert os.path.isfile(CONFIG_PATH), f"Config not found: {CONFIG_PATH}"
assert os.path.isfile(WEIGHTS_PATH), f"Weights not found: {WEIGHTS_PATH}"
assert os.path.isdir(VIDEO_DIR), f"Video dir not found: {VIDEO_DIR}"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# load once
model = load_model(CONFIG_PATH, WEIGHTS_PATH)

# ==== SAM2 导入与初始化 ====
SAM2_AVAILABLE = False
if USE_SAM2:
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        SAM2_AVAILABLE = True
    except Exception as e:
        print("[WARN] SAM2 not available. Set USE_SAM2=False or install SAM2. Error:", e)

def init_sam2_predictor():
    assert SAM2_AVAILABLE, "SAM2 is not available. Please install SAM2 or set USE_SAM2=False."
    assert os.path.isfile(SAM2_CKPT), f"SAM2 checkpoint not found: {SAM2_CKPT}"
    sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT)
    sam2_model.to(device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def color_from_label(label: str) -> tuple:
    """稳定可复现的颜色（BGR）"""
    seed = abs(hash(label)) % (2**32)
    rng = np.random.default_rng(seed)
    c = rng.integers(64, 256, size=3, dtype=np.uint8)
    return int(c[0]), int(c[1]), int(c[2])

# util: parse targets inside [...] in filename
# e.g. "pid199_[climber,pebbles].mp4" -> ["climber","pebbles"]
TARGET_RE = re.compile(r"\[(.+?)\]")

def parse_targets_from_name(name: str):
    m = TARGET_RE.search(name)
    if not m:
        return []
    inside = m.group(1)
    parts = [p.strip().strip('"').strip("'") for p in inside.split(",") if p.strip()]
    return [p for p in parts if p]

def iter_videos(root: str):
    rootp = Path(root)
    for p in sorted(rootp.glob("*.mp4")):
        yield p
    for p in sorted(rootp.glob("**/*.mp4")):
        if p.parent == rootp:
            continue
        yield p

# ==== 坐标转换：cxcywh_rel -> xyxy_rel / xyxy_abs ====
def cxcywh_rel_to_xyxy_rel(cx, cy, w, h):
    x1 = max(0.0, cx - w * 0.5)
    y1 = max(0.0, cy - h * 0.5)
    x2 = min(1.0, cx + w * 0.5)
    y2 = min(1.0, cy + h * 0.5)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def cxcywh_rel_to_xyxy_abs(cx, cy, w, h, W, H):
    x1 = (cx - w * 0.5) * W
    y1 = (cy - h * 0.5) * H
    x2 = (cx + w * 0.5) * W
    y2 = (cy + h * 0.5) * H
    x1 = max(0.0, min(float(x1), W - 1))
    y1 = max(0.0, min(float(y1), H - 1))
    x2 = max(0.0, min(float(x2), W - 1))
    y2 = max(0.0, min(float(y2), H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2
# ==========================================

# main per-video runner
def run_on_video(video_path: Path, out_root: Path):
    video_stem = video_path.stem
    targets = parse_targets_from_name(video_path.name)
    if not targets:
        print(f"[WARN] No targets parsed from: {video_path.name} — skipped.")
        return

    vid_out_dir = out_root / video_stem
    tmp_dir     = vid_out_dir / "tmp_jpg"
    masks_root  = vid_out_dir / MASKS_SUBDIR  # masks/
    vid_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if USE_SAM2:
        masks_root.mkdir(parents=True, exist_ok=True)

    # 为每个目标建立固定的子目录：0_[name], 1_[name], ...
    target_dirs = []
    for i, tgt in enumerate(targets):
        d = masks_root / f"{i}_[{tgt}]"
        d.mkdir(parents=True, exist_ok=True)
        target_dirs.append(d)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing: {video_path.name}")
    print(f"Targets: {targets}")
    idx = 0

    # SAM2 预测器
    sam2_pred = None
    if USE_SAM2:
        if not SAM2_AVAILABLE:
            raise RuntimeError("USE_SAM2=True but SAM2 is not available. Please install SAM2.")
        sam2_pred = init_sam2_predictor()

    first_seg_written = False

    with tqdm(total=N if N > 0 else None, desc=f"{video_stem}") as pbar:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # 写临时 jpg 给 DINO
            fname = f"frame_{idx:06d}.jpg"
            tmp_path = str(tmp_dir / fname)
            cv2.imwrite(tmp_path, frame_bgr)

            # DINO 读取（image_source=RGB）
            image_source, image = load_image(tmp_path)

            # 每个目标收集该帧的相对框（cx,cy,w,h）并转像素XYXY
            boxes_per_target_abs = [[] for _ in targets]
            for t_idx, tgt in enumerate(targets):
                caption = tgt if tgt.endswith(".") else tgt + "."
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=caption,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                )
                if len(phrases) > 0:
                    boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
                    for i_det in range(boxes_np.shape[0]):
                        cx, cy, w, h = map(float, boxes_np[i_det])  # 相对(cx,cy,w,h)
                        # 转绝对xyxy像素
                        x1 = (cx - w * 0.5) * W
                        y1 = (cy - h * 0.5) * H
                        x2 = (cx + w * 0.5) * W
                        y2 = (cy + h * 0.5) * H
                        x1 = max(0.0, min(x1, W - 1))
                        y1 = max(0.0, min(y1, H - 1))
                        x2 = max(0.0, min(x2, W - 1))
                        y2 = max(0.0, min(y2, H - 1))
                        if x2 < x1: x1, x2 = x2, x1
                        if y2 < y1: y1, y2 = y2, y1
                        # 非空框才收
                        if (x2 - x1) > 1e-3 and (y2 - y1) > 1e-3:
                            boxes_per_target_abs[t_idx].append([x1, y1, x2, y2])

            # 是否在本帧做分割
            do_seg = USE_SAM2 and (
                (SEGMENT_ONLY_FIRST_FRAME and idx == 0) or
                ((not SEGMENT_ONLY_FIRST_FRAME) and (idx % SEGMENT_EVERY_N == 0))
            )

            # 每个目标生成一张 union 掩码（无框则全黑）
            if do_seg:
                sam2_pred.set_image(image_source)  # RGB
                union_masks = []
                for t_idx, tgt in enumerate(targets):
                    tgt_mask = np.zeros((H, W), dtype=np.uint8)
                    for b in boxes_per_target_abs[t_idx]:
                        b = np.asarray(b, dtype=np.float32)
                        m, s, _ = sam2_pred.predict(box=b, multimask_output=False)
                        if m.ndim == 3:  # (1,H,W) → (H,W)
                            m = m[0]
                        m_bin = (m > 0).astype(np.uint8)
                        tgt_mask |= m_bin
                    union_masks.append(tgt_mask)

                # 保存每个目标的掩码（masks/0_[name]/frame_xxxxxx.png）
                for t_idx, tgt in enumerate(targets):
                    out_png = target_dirs[t_idx] / f"frame_{idx:06d}.png"
                    cv2.imwrite(str(out_png), union_masks[t_idx] * 255)

                # 第1帧：叠加并保存 first_frame_segmented.jpg
                if not first_seg_written:
                    base_rgb = image_source.copy()
                    overlay_rgb = base_rgb.copy()
                    for t_idx, tgt in enumerate(targets):
                        m = union_masks[t_idx]
                        color_bgr = color_from_label(tgt)
                        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                        colored = np.zeros_like(overlay_rgb, dtype=np.uint8); colored[:] = color_rgb
                        overlay_rgb = np.where(
                            m[..., None] == 1,
                            (MASK_ALPHA_FILL * colored + (1 - MASK_ALPHA_FILL) * overlay_rgb).astype(np.uint8),
                            overlay_rgb
                        )
                        cnts, _ = cv2.findContours((m * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay_rgb, cnts, -1, color_bgr, thickness=2)
                    out_bgr_seg = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(vid_out_dir / "first_frame_segmented.jpg"), out_bgr_seg)
                    first_seg_written = True

            # 清理 tmp
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            idx += 1
            pbar.update(1)

    cap.release()
    print(f"Done: {video_path.name}")
    print("  First-frame (seg):", vid_out_dir / "first_frame_segmented.jpg")
    print("  Masks saved under:", masks_root)



def main():
    out_root = Path(OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = list(iter_videos(VIDEO_DIR))
    if not videos:
        print(f"No mp4 files found in {VIDEO_DIR}")
        return

    for vp in videos:
        # only process files that have [...] targets in the name
        if not TARGET_RE.search(vp.name):
            print(f"[SKIP] no [targets] in filename: {vp.name}")
            continue
        run_on_video(vp, out_root)

if __name__ == "__main__":
    main()
