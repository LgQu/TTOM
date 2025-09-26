import shutil
from pathlib import Path
import re
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=False,
                   default=Path("/scratch/e1351271/video_gen/data/t2v_compbench/4_motion_binding.gpt-4o/mem_s0_ori_wan_enriched_lora32_jsdGs_g5_ls5_i8_[cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o]"))
    p.add_argument("--target-dir", type=Path, required=False,
                   default=Path("/scratch/e1351271/benchmark/T2V-CompBench/data/mem_s0_ori_wan_enriched_lora32_jsdGs_g5_ls5_i8/motion_binding"))
    p.add_argument("--sample", action="store_true",
                   help="启用 sample 模式：只处理 pid-file 中列出的 pid")
    p.add_argument("--pid-file", type=Path, required=False,
                   default=Path("/scratch/e1351271/video_gen/cache/sample_id/cache_4_motion_binding_gpt-4o.txt"),
                   help="sample 模式下使用的 pid 列表文件（每行一个整数 pid）")
    p.add_argument("--max-pid", type=int, default=200,
                   help="全量模式下用于缺失检查的最大 pid（检查范围 0..max_pid-1）")
    return p.parse_args()

def read_required_pids(pid_file: Path):
    with pid_file.open("r") as f:
        return sorted(int(line.strip()) for line in f if line.strip().isdigit())

def main():
    args = parse_args()

    source_dir: Path = args.source_dir
    target_dir: Path = args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # ===== 获取待处理文件列表 / 待处理 pid 集 =====
    if args.sample:
        required_pids = read_required_pids(args.pid_file)
        print(f"[Sample 模式] 从 {args.pid_file} 读取 {len(required_pids)} 个 pid")
        # 只遍历这些 pid，各自匹配 mp4/gif
        media_files = []
        for pid in required_pids:
            media_files += list(source_dir.glob(f"pid{pid}_*.mp4"))
            media_files += list(source_dir.glob(f"pid{pid}_*.gif"))
        expected_pid_set = set(required_pids)
    else:
        # 全量模式：扫描全部
        media_files = sorted(
            list(source_dir.glob("pid*_*.mp4")) +
            list(source_dir.glob("pid*_*.gif"))
        )
        expected_pid_set = set(range(1, 1+args.max_pid))

    # ===== 复制并重命名 =====
    count = 0
    present_pids = set()

    for file in media_files:
        m = re.match(r"pid(\d+)_.*\.(?:mp4|gif)$", file.name, flags=re.IGNORECASE)
        if not m:
            print(f"Skipped file {file.name}: pattern not matched.")
            continue

        pid_num = int(m.group(1))

        # sample 模式下，若 pid 不在列表里则跳过（防御）
        if args.sample and pid_num not in expected_pid_set:
            continue

        present_pids.add(pid_num)
        new_idx = pid_num+1
        new_name = f"{new_idx:04d}{file.suffix.lower()}"
        new_path = target_dir / new_name
        shutil.copy(file, new_path)
        print(f"Copied {file.name} -> {new_name}")
        count += 1

    print(f"\n✅ 总计复制: {count} 个文件")

    # ===== 缺失检查 =====
    missing_pids = sorted(expected_pid_set - present_pids)
    if missing_pids:
        print("\n⚠️  缺少以下 pid 对应的视频文件：")
        for i in range(0, len(missing_pids), 10):
            print(", ".join(f"pid{n}" for n in missing_pids[i:i+10]))

if __name__ == "__main__":
    main()
