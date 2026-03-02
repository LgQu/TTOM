# TTOM: Test-Time Optimization and Memorization for Compositional Video Generation

<p align="center">
  <img src="assets/framework.png" alt="TTOM Framework Overview" width="85%">
</p>

<p align="center">
  <strong>TTOM: Test-Time Optimization and Memorization for Compositional Video Generation</strong>
</p>

<p align="center">
  [ICLR 2026] Official repository
</p>

---

## 🔥 News

- **2026-01**: 🎉 TTOM has been **accepted to ICLR 2026**!
- **2025-12**: 🔥 Released inference code and model weights.

## 🗓️ Todo List

- [x] Release inference code
- [x] Release model weights
- [ ] Release training code

## 📖 Overview

**TTOM** is a test-time optimization and memorization framework for **compositional video generation**. It addresses the challenge of generating videos with multiple objects, attributes, and motions that faithfully follow complex text prompts.

The framework operates in two phases:

1. **Meta Extraction & Layout Generation** (`gen_cache`) – Uses GPT-4o to extract object metadata and generate spatial-temporal layouts from text prompts.
2. **Video Generation with TTOM** (`gen_benchmarks`) – Generates videos using [Wan2.1](https://github.com/Wan-Video/Wan2.1) conditioned on extracted metadata and layouts, with iterative test-time optimization of cross-attention via LoRA.

Built on top of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), an efficient diffusion inference engine.

## 🎥 Qualitative Results

<p align="center">
  <img src="assets/qualitative_t2vcompbench.png" alt="Qualitative Results on T2VCompBench" width="90%">
</p>

<p align="center">
  <img src="assets/mem_qualitative.png" alt="Memorization Qualitative Results" width="90%">
</p>

## 📂 Project Structure

```text
TTOM/
├── generation/
│   ├── gen_cache.py              # Phase 1: Meta extraction and layout generation
│   ├── gen_benchmarks.py         # Phase 2: Video generation with TTOM
│   └── get_attnmap.py            # Attention map generation
├── utils/
│   ├── gdino_detection_video.py  # GroundingDINO object detection + SAM2 segmentation
│   ├── evaluate_miou.py          # mIoU evaluation between attention and masks
│   ├── visualize_attn_maps.py    # Attention map visualization
│   └── ...                       # Other utility functions
├── ttom/                         # Core TTOM implementation
└── scripts/                      # Batch processing scripts
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended ≥ 24 GB VRAM for Wan2.1-T2V-14B)
- `pip` and a virtualenv/conda environment

> 💡 **Note:** [GroundingDINO](#optional-groundingdino--sam2) and [SAM2](#optional-groundingdino--sam2) are **only required** for attention-layout overlap evaluation. For basic video generation, skip them.

### 1. Install TTOM

```bash
git clone https://github.com/LgQu/TTOM.git
cd TTOM
pip install -r requirements.txt
pip install -e .
```

### 2. Download Wan2.1-T2V-14B

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./models/Wan2.1-T2V-14B
```

> 💡 The download can be large. The model will be saved to `./models/Wan2.1-T2V-14B/`.

### 3. Optional: GroundingDINO & SAM2

<details>
<summary>Click to expand (only needed for attention-layout evaluation)</summary>

**GroundingDINO**

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e . && cd ..
```

**SAM2**

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2 && pip install -e . && cd ..
```

</details>

### 4. Configuration

Set up your OpenAI API key for GPT-4o prompt processing:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

## 🚀 Quickstart

### Step 1: Build Layout Cache

```bash
python generation/gen_cache.py \
    --benchmark_source t2vcompbench \
    --benchmark_type 1_consistent_attr \
    --start_idx 0 \
    --end_idx 199 \
    --skip_if_exists
```

**Outputs:**
- `cache/{benchmark_source}_{benchmark_type}-gpt_4o.json` – enriched prompts, object metadata, and layouts.
- `data/layout/boxes_{cache_name}/layout_{pid}.gif` – layout visualization GIFs.

### Step 2: Generate Videos with TTOM

```bash
python generation/gen_benchmarks.py \
    --pid 0 \
    --cache_type t2vcompbench_motion_binding_gpt-4o \
    --guidance_type lora \
    --target_layers [3] \
    --max_iter 8 \
    --max_guidance_step 5 \
    --max_lora_step 5 \
    --target_modules "cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o" \
    --jsd_loss_weight 1.0 \
    --min_loss_value 0.06 \
    --save_lora_weight False \
    --skip_existed_prompt False \
    --prefix "test" \
    --strat_id 0
```

**Outputs:**
- `data/benchmarks/{cache_type}/.../{pid}_{tag}.mp4` – generated videos.

## 🔧 Advanced Usage

### Meta Extraction & Layout Generation

Uses GPT-4o to enrich prompts, extract object instances/attributes, and generate spatial-temporal layouts:

```bash
python generation/gen_cache.py \
    --benchmark_source t2vcompbench \
    --benchmark_type 1_consistent_attr \
    --start_idx 0 \
    --end_idx 199 \
    --skip_if_exists
```

### Video Generation with TTOM Strategies

`gen_benchmarks.py` generates videos using Wan2.1 with TTOM-style test-time optimization and memorization:

```bash
python generation/gen_benchmarks.py \
    --pid 0 \
    --cache_type t2vcompbench_motion_binding_gpt-4o \
    --guidance_type lora \
    --target_layers [3] \
    --max_iter 8 \
    --max_guidance_step 5 \
    --max_lora_step 5 \
    --target_modules "cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o" \
    --jsd_loss_weight 1.0 \
    --com_loss_weight 0.0 \
    --min_loss_value 0.03 \
    --save_lora_weight False \
    --save_mask False \
    --skip_existed_prompt False \
    --prefix "" \
    --strat_id 0
```

<details>
<summary>📋 Full argument reference</summary>

| Argument | Description | Default |
|---|---|---|
| `--pid` | Sample ID to generate | *required* |
| `--cache_type` | Cache file name | `t2vcompbench_motion_binding_gpt-4o` |
| `--guidance_type` | Guidance type: `lora`, `lvd`, `none` | `lora` |
| `--target_layers` | Transformer layers for guidance | `[3]` |
| `--max_iter` | Number of TTOM iterations | `8` |
| `--max_guidance_step` | Max guidance steps per iteration | `5` |
| `--max_lora_step` | Max LoRA update steps per iteration | `5` |
| `--target_modules` | Modules to apply LoRA | `cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o` |
| `--jsd_loss_weight` | JSD loss weight | `1.0` |
| `--com_loss_weight` | Composition loss weight | `0.0` |
| `--min_loss_value` | Min loss value threshold | `0.03` |
| `--save_lora_weight` | Save LoRA weights | `False` |
| `--save_mask` | Save attention masks | `False` |
| `--skip_existed_prompt` | Skip existing outputs | `False` |
| `--prefix` | Output directory prefix | `""` |
| `--strat_id` | TTOM strategy (0=update, 1=load, 2=load+update) | `0` |

</details>

### Attention Map Analysis

<details>
<summary>Click to expand full evaluation pipeline</summary>

**1. Generate attention maps:**

```bash
bash scripts/run_attnmap_batch.sh
```

> 💡 Before running, configure `CONDA_PYTHON` and `BASE_DIR` in the script.

**2. Run GroundingDINO detection + SAM2 segmentation:**

```bash
python utils/gdino_detection_video.py
```

**3. Compute mIoU:**

```bash
python utils/evaluate_miou.py \
    --attn_dir data/attn_maps/wan21_lora/ \
    --dino_dir data/dino_results_batch/ \
    --output_dir data/miou_summary/
```

**4. Visualize attention maps:**

```bash
python utils/visualize_attn_maps.py \
    --pid 0 \
    --step_id 40 \
    --layer_id 3 \
    --inst_id 0 \
    --save
```

</details>

## 📁 Output Structure

```text
data/
├── attn_maps/            # Attention map files (.pt)
├── dino_results_batch/   # GroundingDINO + SAM2 detection/segmentation results
├── miou_summary/         # mIoU evaluation results and summary stats
├── benchmarks/           # Generated videos
└── layout/               # Layout visualizations (GIFs)
```

## 🙏 Acknowledgements

We thank the authors and maintainers of the following projects:

- [Wan2.1](https://github.com/Wan-Video/Wan2.1) – Open and advanced large-scale video generative models.
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) – Efficient diffusion model inference engine.
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) – Open-set object detection with language grounding.
- [SAM2](https://github.com/facebookresearch/segment-anything-2) – Segment Anything 2 for high-quality segmentation.

## ⭐ Citation

If you find TTOM useful, please consider giving this repository a star ⭐ and citing our paper:

```bibtex
@inproceedings{ttom2026iclr,
  title={TTOM: Test-Time Optimization and Memorization for Compositional Video Generation},
  year={2026},
  booktitle={International Conference on Learning Representations (ICLR)}
}
```
