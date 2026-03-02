# TTOM: Test-Time Optimization and Memorization for Compositional Video Generation

Test-time optimization & memorization framework for **compositional video generation** built on top of **Wan2.1** and DiffSynth.  

<p align="center">
  <img src="assets/framework.pdf" alt="TTOM Framework Overview" width="75%">
</p>

---

## News

- 2026-01: TTOM has been **accepted to ICLR 2026**.
- 2025-12: **TTOM** codebase released.

---

## Highlights

- **Compositional video generation**: Benchmarks and pipelines for challenging motion & attribute binding scenarios.
- **Test-time optimization + memorization**: Iteratively refines cross-attention and LoRA parameters at inference time.
- **Layout-aware generation**: Uses GPT-4o to extract object metadata and generate spatial-temporal layouts.
- **Attention-layout evaluation**: End-to-end pipeline for attention map extraction, GroundingDINO + SAM2 segmentation, and mIoU analysis.
- **Built on Wan2.1 & DiffSynth**: Reuses efficient Wan2.1 video backbone and DiffSynth engineering stack ([Wan2.1](https://github.com/Wan-Video/Wan2.1), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)).

<p align="center">
  <img src="assets/qualitative_t2vcompbench.pdf" alt="Qualitative Results on T2VCompBench" width="90%">
</p>

---

## Table of Contents

- [TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](#ttom-test-time-optimization-and-memorization-for-compositional-video-generation)
  - [News](#news)
  - [Highlights](#highlights)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install TTOM](#install-ttom)
    - [Download Wan2.1 model](#download-wan21-model)
    - [Optional: GroundingDINO \& SAM2](#optional-groundingdino--sam2)
  - [Configuration](#configuration)
  - [Quickstart](#quickstart)
    - [Step 1: Build layout cache (`gen_cache`)](#step-1-build-layout-cache-gen_cache)
    - [Step 2: Generate videos (`gen_benchmarks`)](#step-2-generate-videos-gen_benchmarks)
  - [Advanced Usage](#advanced-usage)
    - [Meta extraction \& layout generation](#meta-extraction--layout-generation)
    - [Video generation with TTOM strategies](#video-generation-with-ttom-strategies)
    - [Attention map analysis](#attention-map-analysis)
  - [Output Structure](#output-structure)
  - [Acknowledgements](#acknowledgements)

---

## Overview

TTOM is a framework for compositional video generation that consists of two main phases:

1. **Meta Extraction and Layout Generation** (`gen_cache`) – Uses GPT-4o to extract object metadata and generate spatial layouts from prompts.
2. **Video Generation** (`gen_benchmarks`) – Generates videos using Wan2.1 conditioned on the extracted metadata and layouts with test-time optimization.

Additionally, the framework includes **attention map analysis** (`get_attnmap.py` and evaluation utilities) for quantifying attention-layout overlap between cross-modal attention maps and segmentation maps.

---

## Project Structure

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

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended ≥ 24 GB VRAM for Wan2.1-T2V-14B)
- `pip` and a virtualenv/conda environment

> **Note:** Steps 3 and 4 below are **only required** if you want to perform attention-layout overlap analysis using GroundingDINO detection and SAM2 segmentation.  
> For **basic video generation**, only step 1 and 2 are required.

TTOM is built on top of **DiffSynth**, an efficient diffusion inference engine.  
For more information about DiffSynth, see: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

### Install TTOM

```bash
pip install -r requirements.txt
pip install -e .
```

### Download Wan2.1 model

Download the **Wan2.1-T2V-14B** model from Hugging Face into the `models` directory:

```bash
# Install huggingface_hub if not already installed
pip install "huggingface_hub[cli]"

# Download the model
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./models/Wan2.1-T2V-14B
```

> The download can be large and may take some time.  
> The model will be saved to: `./models/Wan2.1-T2V-14B/`.

### Optional: GroundingDINO & SAM2

These components are used **only for attention-layout overlap evaluation**.

**GroundingDINO**

```bash
cd TTOM
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

**SAM2**

```bash
cd TTOM
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2
pip install -e .
```

---

## Configuration

Set up your OpenAI API key for GPT-4o prompt processing:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

---

## Quickstart

This section shows how to:

1. Build a **layout cache** using GPT-4o.  
2. Generate **videos** from the cache with TTOM.

### Step 1: Build layout cache (`gen_cache`)

```bash
python generation/gen_cache.py \
    --benchmark_source t2vcompbench \
    --benchmark_type 1_consistent_attr \
    --start_idx 0 \
    --end_idx 199 \
    --skip_if_exists
```

**Outputs**

- `cache/{benchmark_source}_{benchmark_type}-gpt_4o.json`  
  JSON cache with enriched prompts, object metadata, and spatial-temporal layouts.
- `data/layout/boxes_{cache_name}/layout_{pid}.gif`  
  GIF visualizations showing object positions over time for each prompt.

### Step 2: Generate videos (`gen_benchmarks`)

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

**Typical outputs**

- `data/benchmarks/{cache_type}/{prefix}_wan_enriched_lora32_jsdGs_g{max_guidance_step}_ls{max_lora_step}_i{max_iter}_[{target_modules}]/pid{pid}_{tag}.mp4`  
  Generated videos for each `pid`.

---

## Advanced Usage

### Meta extraction & layout generation

This stage uses GPT-4o to:

- Enrich prompts.
- Extract object instances and attributes.
- Generate spatial-temporal layouts.

Run:

```bash
python generation/gen_cache.py \
    --benchmark_source t2vcompbench \
    --benchmark_type 1_consistent_attr \
    --start_idx 0 \
    --end_idx 199 \
    --skip_if_exists
```

Key behaviors:

- **Prompt enrichment** with GPT-4o.  
- **Object metadata extraction** (instances, attributes, relations).  
- **Layout generation** describing positions over time.

### Video generation with TTOM strategies

`gen_benchmarks.py` generates videos using Wan2.1 with TTOM-style test-time optimization and memorization.

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

Key arguments:

- `--pid`: Sample ID to generate (required).  
- `--cache_type`: Cache name, e.g. `t2vcompbench_motion_binding_gpt-4o`.  
- `--guidance_type`: `"lora"`, `"lvd"`, or `"none"`.  
- `--target_layers`: List of transformer layers for applying guidance.  
- `--max_iter`: Number of TTOM iterations.  
- `--max_guidance_step`, `--max_lora_step`: Steps for guidance and LoRA updates.  
- `--target_modules`: Modules to apply LoRA (e.g. cross-attention q/k/v/o).  
- `--jsd_loss_weight`, `--com_loss_weight`: Loss weights for TTOM objectives.  
- `--strat_id`: TTOM strategy (0 = update, 1 = load, 2 = load+update).

### Attention map analysis

This pipeline quantifies how well attention aligns with detected objects and layouts.

1. **Generate attention maps** (using prompts from `TTOM/cache/cache_train_motion_gpt-4o.json`):

   ```bash
   # Note: Before running, configure CONDA_PYTHON and BASE_DIR in the script
   bash scripts/run_attnmap_batch.sh
   ```

   Outputs:

   - `data/attn_maps/wan2.1-t2v-14b/pid{pid}_insts{insts_str}.pt`
   - `data/attn_maps/wan21_lora/{cache_type}/pid{pid}_insts{insts_str}.pt`

2. **Run GroundingDINO detection + SAM2 segmentation**:

   ```bash
   python utils/gdino_detection_video.py
   ```

   This script:

   - Detects objects in generated videos using GroundingDINO.  
   - Performs segmentation with SAM2.  
   - Saves detection results and masks.

   Outputs:

   - `data/dino_results_batch/{video_name}/` – Detection results with bounding boxes and scores.  
   - `data/dino_results_batch/{video_name}/masks/` – Segmentation masks (PNG).

3. **Compute mIoU between attention and masks**:

   ```bash
   python utils/evaluate_miou.py \
       --attn_dir data/attn_maps/wan21_lora/ \
       --dino_dir data/dino_results_batch/ \
       --output_dir data/miou_summary/
   ```

   Outputs:

   - `data/miou_summary/` – mIoU evaluation results and statistics.  
   - Plots and metrics for attention-layout overlap.

4. **Visualize attention maps and detections**:

   ```bash
   python utils/visualize_attn_maps.py \
       --pid 0 \
       --step_id 40 \
       --layer_id 3 \
       --inst_id 0 \
       --save
   ```

   Outputs:

   - `data/attention_visualizations/` – Heatmap visualizations of attention maps.  
   - Key-frame attention analysis plots.

---

## Output Structure

```text
data/
├── attn_maps/            # Attention map files (.pt)
├── dino_results_batch/   # GroundingDINO + SAM2 detection/segmentation results
├── miou_summary/         # mIoU evaluation results and summary stats
├── benchmarks/           # Generated videos
└── layout/               # Layout visualizations (e.g., GIFs)
```

---

## Acknowledgements

TTOM builds upon and is inspired by the following excellent open-source projects:

- **Wan2.1** – Open and advanced large-scale video generative models. See [Wan2.1 repository](https://github.com/Wan-Video/Wan2.1).  
- **DiffSynth-Studio** – Efficient diffusion model inference engine.  
- **GroundingDINO** – Open-set object detection with language grounding.  
- **SAM2** – Segment Anything 2 for high-quality segmentation.  
- **OpenAI GPT-4o** – Used for prompt enrichment and metadata extraction.

We thank the authors and maintainers of these projects for making their work publicly available.

