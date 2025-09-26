# TTOM: Test-Time Optimization and Memorization for Compositional Video Generation

ICLR'26 submission 3050: TTOM: Test-Time Optimization and Memorization for Compositional Video Generation

## Overview

TTOM is a framework for compositional video generation that consists of two main phases:

1. **Meta Extraction and Layout Generation** (`gen_cache`) - Uses GPT-4o to extract object metadata and generate spatial layouts
2. **Video Generation** (`gen_benchmarks`) - Generates videos using the extracted metadata and layouts

Additionally, the framework includes attention map analysis (`genattnmap`) for evaluating attention-layout overlap between cross-modal attention maps and segmentation maps.

## Configuration

Set up API keys:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Project Structure

```
TTOM/
├── generation/
│   ├── gen_cache.py          # Phase 1: Meta extraction and layout generation
│   ├── gen_benchmarks.py     # Phase 2: Video generation
│   └── get_attnmap.py        # Attention map generation
├── utils/
│   ├── gdino_detection_video.py  # GroundingDINO object detection
│   ├── evaluate_miou.py          # mIoU evaluation
│   ├── visualize_attn_maps.py    # Attention map visualization
│   └── ...                      # Other utility functions
├── ttom/                       # Core TTOM implementation
└── scripts/                    # Batch processing scripts
```

## Installation

### Prerequisites

**Note:** Steps 3 and 4 are only required if you want to perform attention-layout overlap analysis using GroundingDINO detection and SAM2 segmentation. For basic video generation, only step 1 is needed.

**Based on DiffSynth:** This project is built on top of DiffSynth, an efficient diffusion model inference engine. For more information about DiffSynth, please visit the official repository: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)


1. **Install TTOM Dependencies** (Required)
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Download wan2.1-t2v-14b Model**

   Download the wan2.1-t2v-14b model from Hugging Face to the `models` directory:

   ```bash
   # Install huggingface_hub if not already installed
   pip install "huggingface_hub[cli]"
   
   # Download the model
   huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./models/Wan2.1-T2V-14B
   ```

   **Note:** The download may take some time depending on your internet connection speed. The model will be saved to `./models/Wan2.1-T2V-14B/`.

3. **GroundingDINO Installation** (Optional - Only needed for attention-layout analysis)
   ```bash
   cd TTOM
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   cd GroundingDINO
   pip install -e .
   ```

4. **SAM2 Installation** (Optional - Only needed for attention-layout analysis)
   ```bash
   cd TTOM
   git clone https://github.com/facebookresearch/segment-anything-2.git sam2
   cd sam2
   pip install -e .
   ```


## Usage

### Phase 1: Meta Extraction and Layout Generation

This phase uses GPT-4o to extract object metadata and generate spatial layouts from text prompts.

```bash
python generation/gen_cache.py \
    --benchmark_source t2vcompbench \
    --benchmark_type 1_consistent_attr \
    --start_idx 0 \
    --end_idx 199 \
    --skip_if_exists
```

**Key Features:**
- Prompt enrichment using GPT-4o
- Object metadata extraction
- Layout generation with spatial reasoning

**Output:**
- `cache/{benchmark_source}_{benchmark_type}-gpt_4o.json` - JSON cache with enriched prompts, object metadata, and layouts
- `data/layout/boxes_{cache_name}/layout_{pid}.gif` - Layout visualization GIFs showing object positions over time


### Phase 2: Video Generation

This phase generates videos using the extracted metadata and layouts.

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

**Parameter Descriptions:**
- `--pid`: Sample ID to generate (required)
- `--cache_type`: Name of the cache file (default: "t2vcompbench_motion_binding_gpt-4o")
- `--guidance_type`: Type of guidance ("lora", "lvd", "none")
- `--target_layers`: List of target layers for guidance (default: [3])
- `--max_iter`: Maximum number of iterations (default: 8)
- `--max_guidance_step`: Maximum guidance steps (default: 5)
- `--max_lora_step`: Maximum LoRA steps (default: 5)
- `--target_modules`: Target modules for LoRA (default: "cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o")
- `--jsd_loss_weight`: Jensen-Shannon divergence loss weight (default: 1.0)
- `--com_loss_weight`: Composition loss weight (default: 0.0)
- `--min_loss_value`: Minimum loss value threshold (default: 0.03)
- `--save_lora_weight`: Whether to save LoRA weights (default: False)
- `--save_mask`: Whether to save attention masks (default: False)
- `--skip_existed_prompt`: Skip if output already exists (default: False)
- `--prefix`: Output directory prefix (default: "")
- `--strat_id`: Strategy ID for TTOM (0: update, 1: load, 2: load+update)

**Key Features:**
- LoRA-based guidance for object-aware generation
- Attention map extraction and analysis
- Multiple guidance strategies
- Batch processing support

**Output:**
- `data/benchmarks/{cache_type}/{prefix}_wan_enriched_lora32_jsdGs_g{max_guidance_step}_ls{max_lora_step}_i{max_iter}_[{target_modules}]/pid{pid}_{tag}.mp4` - Generated videos
- `data/attn_maps/wan21_lora/{cache_type}/pid{pid}_insts{insts_str}.pt` - Attention map files (when save_attn_map=True)

### * Attention Map Analysis

1. Generate attention maps for analysis of attention-layout overlap (using prompts from TTOM/cache/cache_train_motion_gpt-4o.json):

```bash
# Generate attention maps for multiple PIDs
# Note: Before running, configure CONDA_PYTHON and BASE_DIR in the script
bash scripts/run_attnmap_batch.sh
```



**Output:**
- `data/attn_maps/wan2.1-t2v-14b/pid{pid}_insts{insts_str}.pt` - Attention map files containing cross attention data


2. **Run GroundingDINO Detection**
   ```bash
   python utils/gdino_detection_video.py
   ```
   
   This script:
   - Detects objects in generated videos using GroundingDINO
   - Performs segmentation using SAM2
   - Saves detection results and masks

   **Output:**
   - `data/dino_results_batch/{video_name}/` - Detection results with bounding boxes and confidence scores
   - `data/dino_results_batch/{video_name}/masks/` - Segmentation masks (PNG files)

3. **Calculate mIoU**
   ```bash
   python utils/evaluate_miou.py \
       --attn_dir data/attn_maps/wan21_lora/ \
       --dino_dir data/dino_results_batch/ \
       --output_dir data/miou_summary/
   ```

   **Output:**
   - `data/miou_summary/` - mIoU evaluation results and statistics
   - Attention-layout overlap analysis plots and metrics

4. Visualize attention maps and detection results:

```bash
python utils/visualize_attn_maps.py \
    --pid 0 \
    --step_id 40 \
    --layer_id 3 \
    --inst_id 0 \
    --save
```

**Output:**
- `data/attention_visualizations/` - Heatmap visualizations of attention maps
- Key frame attention analysis plots and statistics

## Output Structure

```
data/
├── attn_maps/           # Attention map files (.pt)
├── dino_results_batch/ # GroundingDINO detection results
├── miou_summary/        # mIoU evaluation results
├── benchmarks/          # Generated videos
└── layout/             # Layout visualizations
```
