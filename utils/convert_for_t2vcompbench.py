import shutil
from pathlib import Path
import re
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=False,
                   default=Path("data/t2v_compbench/4_motion_binding.gpt-4o/mem_s0_ori_wan_enriched_lora32_jsdGs_g5_ls5_i8_[cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o]"))
    p.add_argument("--target-dir", type=Path, required=False,
                   default=Path("benchmark/T2V-CompBench/data/mem_s0_ori_wan_enriched_lora32_jsdGs_g5_ls5_i8/motion_binding"))
    p.add_argument("--sample", action="store_true",
                   help="Enable sample mode: only process pids listed in pid-file")
    p.add_argument("--pid-file", type=Path, required=False,
                   default=Path("cache/sample_id/cache_4_motion_binding_gpt-4o.txt"),
                   help="PID list file used in sample mode (one integer pid per line)")
    p.add_argument("--max-pid", type=int, default=200,
                   help="Maximum pid for missing check in full mode (check range 0..max_pid-1)")
    return p.parse_args()

def read_required_pids(pid_file: Path):
    with pid_file.open("r") as f:
        return sorted(int(line.strip()) for line in f if line.strip().isdigit())

def main():
    args = parse_args()

    source_dir: Path = args.source_dir
    target_dir: Path = args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # ===== Get file list to process / pid set to process =====
    if args.sample:
        required_pids = read_required_pids(args.pid_file)
        print(f"[Sample mode] Read {len(required_pids)} pids from {args.pid_file}")
        # Only iterate these pids, match mp4/gif for each
        media_files = []
        for pid in required_pids:
            media_files += list(source_dir.glob(f"pid{pid}_*.mp4"))
            media_files += list(source_dir.glob(f"pid{pid}_*.gif"))
        expected_pid_set = set(required_pids)
    else:
        # Full mode: scan all
        media_files = sorted(
            list(source_dir.glob("pid*_*.mp4")) +
            list(source_dir.glob("pid*_*.gif"))
        )
        expected_pid_set = set(range(1, 1+args.max_pid))

    # ===== Copy and rename =====
    count = 0
    present_pids = set()

    for file in media_files:
        m = re.match(r"pid(\d+)_.*\.(?:mp4|gif)$", file.name, flags=re.IGNORECASE)
        if not m:
            print(f"Skipped file {file.name}: pattern not matched.")
            continue

        pid_num = int(m.group(1))

        # In sample mode, skip if pid not in list (defensive)
        if args.sample and pid_num not in expected_pid_set:
            continue

        present_pids.add(pid_num)
        new_idx = pid_num+1
        new_name = f"{new_idx:04d}{file.suffix.lower()}"
        new_path = target_dir / new_name
        shutil.copy(file, new_path)
        print(f"Copied {file.name} -> {new_name}")
        count += 1

    print(f"\n✅ Total copied: {count} files")

    # ===== Missing check =====
    missing_pids = sorted(expected_pid_set - present_pids)
    if missing_pids:
        print("\n⚠️  Missing video files for the following pids:")
        for i in range(0, len(missing_pids), 10):
            print(", ".join(f"pid{n}" for n in missing_pids[i:i+10]))

if __name__ == "__main__":
    main()
