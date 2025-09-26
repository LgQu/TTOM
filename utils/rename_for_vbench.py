#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import shutil
from pathlib import Path

# Match pid in filenames like "...pid_1234..."
PID_RE = re.compile(r"pid(\d+)_", re.IGNORECASE)

# benchmark_type = "human_action"
# benchmark_type = "multiple_objects"
# benchmark_type = "object_class"
# benchmark_type = "overall_consistency"
benchmark_type = "scene"
# benchmark_type = "spatial_relationship"
# benchmark_type = "subject_consistency"
# benchmark_type = "temporal_flickering"
# benchmark_type = "temporal_style"

# Default output directory (can be overridden by --out)
DEFAULT_OUT_DIR = f"/scratch/e1351271/benchmark/VBench/vbench_videos/wan21_lora/{benchmark_type}"
DEFAULT_IN_DIR = f"/scratch/e1351271/video_gen/ttt-lm/data/benchmarks/vbench_{benchmark_type}-gpt_4o/mem_s0_ori_wan_enriched_lora32_jsdGs_g5_ls5_i8_[cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o]"
DEFAULT_JSON = f"/scratch/e1351271/video_gen/ttt-lm/cache/vbench_{benchmark_type}-gpt_4o.json"


def safe_filename(s: str) -> str:
    """
    Keep original phrasing (spaces, commas, etc.), only minimal sanitization:
    - Replace / and \ with full-width versions to avoid path breaks
    - Normalize newlines to spaces
    - Strip leading/trailing spaces
    - Remove null chars
    - Truncate overly long names
    """
    if s is None:
        s = ""
    s = s.replace("/", "／").replace("\\", "＼")
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    s = s.replace("\x00", "")
    s = s.strip()
    if not s:
        s = "untitled"
    return s

def find_pid(path: Path) -> str | None:
    m = PID_RE.search(path.name)
    return m.group(1) if m else None

def main():
    parser = argparse.ArgumentParser(description="Copy & rename VBench videos using original_prompt by pid (non-destructive).")
    parser.add_argument("--root",
                        default=DEFAULT_IN_DIR,
                        help=f"Root dir to traverse (default: {DEFAULT_IN_DIR})")
    parser.add_argument("--json",
                        default=DEFAULT_JSON,
                        help=f"JSON file (default: {DEFAULT_JSON})")
    parser.add_argument("--exts", nargs="+", default=[".mp4", ".webm", ".mov", ".mkv"],
                        help="Video extensions to include")
    parser.add_argument("--apply", action="store_true",
                        help="Actually COPY files (omit for dry-run)")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help=f"Target output directory (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] Root not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Cannot create output dir {out_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.json, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Normalize mapping keys to strings
    mapping = {str(k): v for k, v in mapping.items()}

    total = 0
    copied = 0
    skipped = 0
    collisions = 0
    missing_pid = 0
    missing_prompt = 0

    exts_set = {e.lower() for e in args.exts}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts_set:
            continue

        total += 1
        pid = find_pid(p)
        if pid is None:
            print(f"[SKIP] No pid in name: {p}")
            missing_pid += 1
            skipped += 1
            continue

        rec = mapping.get(pid)
        if rec is None:
            print(f"[SKIP] pid {pid} not found in JSON for file: {p}")
            missing_prompt += 1
            skipped += 1
            continue

        original_prompt = rec.get("original_prompt")
        if not original_prompt:
            print(f"[SKIP] pid {pid} has no original_prompt in JSON: {p}")
            missing_prompt += 1
            skipped += 1
            continue

        # Build filename preserving original prompt phrasing; append "-0" before extension
        base_name = safe_filename(original_prompt)
        new_name = f"{base_name}-0{p.suffix}"

        target = out_dir / new_name

        # Deduplicate using "name (2).ext" pattern
        if target.exists():
            base = target.stem
            ext = target.suffix
            i = 2
            dedup = out_dir / f"{base} ({i}){ext}"
            while dedup.exists() and i < 1000:
                i += 1
                dedup = out_dir / f"{base} ({i}){ext}"
            if dedup.exists():
                print(f"[SKIP] Collision not resolved for: {p} -> {target}")
                collisions += 1
                skipped += 1
                continue
            target = dedup

        mode = "[APPLY-COPY]" if args.apply else "[DRY-RUN]"
        print(f"{mode:<12} {p}  ->  {target}")

        if args.apply:
            try:
                shutil.copy2(p, target)
                copied += 1
            except Exception as e:
                print(f"[ERROR] Failed to copy {p} -> {target}: {e}", file=sys.stderr)
                skipped += 1

    print("\n=== Summary ===")
    print(f"Scanned files : {total}")
    print(f"Copied        : {copied}")
    print(f"Skipped       : {skipped}")
    print(f"Missing pid   : {missing_pid}")
    print(f"Missing prompt: {missing_prompt}")
    print(f"Collisions    : {collisions}")
    print(f"Mode          : {'APPLY-COPY' if args.apply else 'DRY-RUN'}")
    print(f"Output dir    : {out_dir}")

if __name__ == "__main__":
    main()
