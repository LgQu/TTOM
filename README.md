# TTOM-CogVideoX: Test-Time Optimization and Memorization for CogVideoX

CogVideoX branch of **[TTOM](https://github.com/LgQu/TTOM)** — Test-Time Optimization and Memorization for Compositional Video Generation.

This branch implements TTOM-style layout-guided backward guidance on **CogVideoX** (5B) models, enabling compositional video generation with multiple objects, attributes, and motions.

## Overview

TTOM-CogVideoX operates in two phases:

1. **Meta Extraction & Layout Generation** — Uses GPT-4o to extract object metadata and generate spatial-temporal layouts from text prompts.
2. **Video Generation with Guidance** — Generates videos using CogVideoX with iterative test-time optimization of cross-attention via latent guidance.

Key features:
- **Layout-guided backward guidance** on CogVideoX latents
- **AMF (Attention Motion Flow)** loss for motion-aware generation
- **JSD, CIoU, region, and COM** losses for spatial accuracy
- Support for CogVideoX-2B and CogVideoX-5B

## Project Structure

```
TTOM-CogVideoX/
├── generation/
│   ├── gen_cache.py              # Phase 1: Meta extraction and layout generation
│   ├── gen_benchmarks.py         # Phase 2: Video generation with guidance
│   └── get_attnmap.py            # Attention map generation
├── dsl/
│   ├── generation/               # Core generation modules
│   │   ├── lvd_cogvideo.py       # LVD-guided CogVideoX generation
│   │   ├── cogvideo.py           # Baseline CogVideoX generation
│   │   └── prompt_parser.py      # Prompt parsing utilities
│   ├── guidance/                 # Guidance pipeline
│   │   ├── guidance_pipeline.py  # LatentGuidance: backward gradient optimization
│   │   ├── amf_loss.py           # Attention Motion Flow loss
│   │   └── energy_functions.py   # Spatial/layout energy functions
│   ├── model/
│   │   └── cogvideox/            # Custom CogVideoX transformer + pipeline
│   └── utils/                    # Layout, cache, prompt utilities
├── ttom/                         # Core TTOM module
│   ├── __init__.py
│   └── cogvideo_guidance.py      # CogVideoX-specific guidance wrapper
├── utils/
│   ├── visualize_attn_maps.py    # Attention map visualization
│   └── evaluate_miou.py          # mIoU evaluation
├── scripts/                      # Batch processing scripts
│   ├── run_benchmarks_batch.sh
│   └── run_attnmap_batch.sh
├── cache/                        # LLM layout caches and prompt files
├── config/                       # YAML configuration files
├── setup.py
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended >= 24 GB VRAM for CogVideoX-5B)

### 1. Install

```bash
git clone https://github.com/LgQu/TTOM.git -b cogvideo
cd TTOM
pip install -r requirements.txt
pip install -e .
```

### 2. Download CogVideoX-5B

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download THUDM/CogVideoX-5b --local-dir ./models/CogVideoX-5b
```

### 3. Configuration

Set up your OpenAI API key for GPT-4o prompt processing:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

## Quickstart

### Step 1: Build Layout Cache

```bash
python generation/gen_cache.py \
    --benchmark_source t2vcompbench \
    --benchmark_type 4_motion_binding \
    --start_idx 0 \
    --end_idx 199 \
    --skip_if_exists
```

**Outputs:**
- `cache/cache_{benchmark_type}_gpt-4o.json` — enriched prompts, object metadata, and layouts.
- `data/layout/boxes_{cache_name}/layout_{pid}.gif` — layout visualization GIFs.

### Step 2: Generate Videos with Guidance

```bash
python generation/gen_benchmarks.py \
    --pid 0 \
    --cache_type 4_motion_binding_gpt-4o \
    --model_path THUDM/CogVideoX-5b \
    --guidance_type main \
    --guidance_attn_strategy middle_2 \
    --guidance_start_step 0 \
    --guidance_end_step 10 \
    --max_iter 5 \
    --lr 5e-3 \
    --set_amf_loss \
    --seed 42
```

**Outputs:**
- `data/benchmarks/{cache_type}/.../pid_{timestamp}.mp4` — generated videos.

## Advanced Usage

### Full Argument Reference

| Argument | Description | Default |
|---|---|---|
| `--pid` | Sample ID to generate | *required* |
| `--cache_type` | Cache file identifier | `4_motion_binding_gpt-4o` |
| `--model_path` | CogVideoX model path | `THUDM/CogVideoX-5b` |
| `--guidance_type` | `main` (with guidance) or `none` | `main` |
| `--guidance_attn_strategy` | Attention layer strategy | `middle_2` |
| `--guidance_start_step` | First guided denoising step | `0` |
| `--guidance_end_step` | Last guided denoising step | `10` |
| `--max_iter` | Max guidance iterations per step | `5` |
| `--lr` | Learning rate for latent optimization | `5e-3` |
| `--set_amf_loss` | Enable AMF loss | `False` |
| `--loss_threshold` | Min loss to stop optimization | `12.0` |
| `--seed` | Random seed | `42` |
| `--skip_existed_prompt` | Skip existing outputs | `False` |
| `--prefix` | Output directory prefix | `""` |

### Attention Map Analysis

**1. Generate attention maps:**

```bash
bash scripts/run_attnmap_batch.sh
```

**2. Visualize attention maps:**

```bash
python utils/visualize_attn_maps.py \
    --pid 0 \
    --step_id 4 \
    --layer_id 20 \
    --inst_id 0 \
    --save
```

**3. Compute mIoU:**

```bash
python utils/evaluate_miou.py \
    --attn_dir data/attn_maps/cogvideox \
    --cache_type 4_motion_binding_gpt-4o \
    --output_dir data/miou_summary \
    --pid 0
```

## Output Structure

```
data/
├── attn_maps/            # Attention map files (.pt)
├── benchmarks/           # Generated videos
├── layout/               # Layout visualizations (GIFs)
└── miou_summary/         # mIoU evaluation results
```

## Acknowledgements

- [TTOM](https://github.com/LgQu/TTOM) — Test-Time Optimization and Memorization framework
- [CogVideoX](https://github.com/THUDM/CogVideo) — Open-source video generation model
- [diffusers](https://github.com/huggingface/diffusers) — Diffusion model inference library

## Citation

```bibtex
@article{qu2025ttom,
  title   = {TTOM: Test-Time Optimization and Memorization for Compositional Video Generation},
  author  = {Leigang Qu and Ziyang Wang and Na Zheng and Wenjie Wang and Liqiang Nie and Tat-Seng Chua},
  journal = {arXiv preprint arXiv:2510.07940},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.07940}
}
```
