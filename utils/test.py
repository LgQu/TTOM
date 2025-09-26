#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

def find_results_jsons(root: Path) -> List[Path]:
    """
    Find all .../miou/inst*/results.json under root.
    """
    matches = []
    # Walk is faster than multiple globs and handles huge trees
    for dirpath, dirnames, filenames in os.walk(root):
        if "miou" not in Path(dirpath).parts:
            continue
        if "results.json" in filenames:
            p = Path(dirpath) / "results.json"
            # enforce inst*/results.json pattern
            if p.parent.name.startswith("inst"):
                matches.append(p)
    return sorted(matches)

def load_mean_soft_iou_per_layer(jpath: Path) -> Dict[int, float]:
    """
    Load results.json and extract mean_soft_iou for each layer.
    Returns a dict {layer_id: value}.
    """
    with jpath.open("r") as f:
        data = json.load(f)
    out = {}
    layers = data.get("layers", {})
    for k, v in layers.items():
        try:
            lid = int(k)
        except ValueError:
            continue
        if "mean_soft_iou" in v:
            out[lid] = float(v["mean_soft_iou"])
    return out

def make_column_name(root: Path, jpath: Path) -> str:
    """
    Create a compact, unique column name based on relative path:
    e.g., pid0_[red kayak]/miou/inst0 -> pid0_[red kayak]/inst0
    """
    rel = jpath.parent  # .../miou/instX
    # strip up to the batch root
    rel_parts = rel.relative_to(root).parts
    # try to drop the 'miou' segment for readability
    if "miou" in rel_parts:
        parts = [p for p in rel_parts if p != "miou"]
    else:
        parts = list(rel_parts)
    # keep only the last two meaningful parts if path is long
    if len(parts) >= 2:
        name = "/".join(parts[-2:])  # e.g., pid0_[red kayak]/inst0
    else:
        name = "/".join(parts)
    return name

def main():
    ap = argparse.ArgumentParser(description="Aggregate mean_soft_iou per layer into CSV.")
    ap.add_argument("--root", required=True,
                    help="Root dir: /scratch/e1351271/video_gen/ttt-lm/data/dino_results_batch")
    ap.add_argument("--out", default="mean_soft_iou_layers.csv",
                    help="Output CSV path")
    ap.add_argument("--num_layers", type=int, default=40,
                    help="Number of layers (rows). Default: 40")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    results = find_results_jsons(root)
    if not results:
        raise SystemExit(f"No results.json found under {root}")

    # Prepare DataFrame with layer ids as index (0..num_layers-1)
    index = list(range(args.num_layers))
    df = pd.DataFrame(index=index)

    for j in results:
        col_name = make_column_name(root, j)
        layer2val = load_mean_soft_iou_per_layer(j)
        col = np.full((args.num_layers,), np.nan, dtype=float)
        for lid, val in layer2val.items():
            if 0 <= lid < args.num_layers:
                col[lid] = val
        df[col_name] = col

    # Nice index name
    df.index.name = "layer_id"
    df.to_csv(args.out, float_format="%.10f")
    print(f"Wrote CSV with shape {df.shape} to {args.out}")
    print(f"Columns (instances): {len(df.columns)}")

if __name__ == "__main__":
    main()
