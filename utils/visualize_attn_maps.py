#!/usr/bin/env python3
"""
Key frames visualization of attention maps (30x52) as heatmap grid
Shows first, last, and 3 evenly sampled middle frames from 21-frame sequences
Supports single layer or multiple layer averaging via configuration file

Usage: 
  python visualize_keyframes.py [--pid PID] [--step_id STEP_ID] [--layer_id LAYER_ID] [--inst_id INST_ID] [--save] [--stats]
  python visualize_keyframes.py --config config.txt [--pid PID] [--save] [--stats]
  python visualize_keyframes.py --full [--pid PID] [--step_id STEP_ID] [--inst_id INST_ID] [--save] [--stats]

Configuration file format:
  step_id=40
  layer_ids=13,15,17
  inst_id=0
"""
# --- Add at the top imports ---
from matplotlib.backends.backend_pdf import PdfPages

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
import re

# --- Add this helper next to parse_layer_ids ---
def parse_frame_indices(value, total_frames=None):
    """
    Parse frame indices from:
      - Comma- or space-separated: "0,4,8" or "0 4 8"
      - Range: "1-5" -> [1,2,3,4,5]
      - Mixed: "0,3-5,10"
    Returns 0-indexed integers. If total_frames is given, filters out-of-range.
    """
    value = value.strip()
    parts = re.split(r'[, ]+', value)
    indices = []
    for part in parts:
        if not part:
            continue
        if '-' in part:
            try:
                a, b = map(int, part.split('-'))
                rng = range(a, b + 1) if a <= b else range(b, a + 1)
                indices.extend(rng)
            except ValueError:
                print(f"Warning: Invalid frame range '{part}', skipping")
        else:
            try:
                indices.append(int(part))
            except ValueError:
                print(f"Warning: Invalid frame index '{part}', skipping")
    # Convert to 0-indexed if user gave 1-based (heuristic: if any index == total_frames, user likely used 1-based)
    # You can comment this out if you strictly use 0-based.
    # Here we accept user’s natural 1-based input: convert iff any index == 1 or all indices >= 1.
    if indices and min(indices) >= 1:
        zero_based = [i - 1 for i in indices]
    else:
        zero_based = indices
    # Dedup & sort
    zero_based = sorted(set(zero_based))
    # Filter by total_frames if provided
    if total_frames is not None:
        zero_based = [i for i in zero_based if 0 <= i < total_frames]
    return zero_based

# --- Add these new functions somewhere above main() ---

def export_frames_as_pdf(frames, frame_indices, save_dir, base_name,
                         single_pdf=True, cmap='viridis', share_cmap=True):
    """
    Export selected frames as PDF(s): either one multi-page PDF (single_pdf=True)
    or one PDF per frame (single_pdf=False). No axes/labels/titles.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Shared color scale across selected frames for comparability
    if share_cmap:
        data = np.stack([frames[i] for i in frame_indices], axis=0)
        vmin, vmax = float(np.min(data)), float(np.max(data))
    else:
        vmin = vmax = None

    if single_pdf:
        pdf_path = os.path.join(save_dir, f"{base_name}.pdf")
        with PdfPages(pdf_path) as pdf:
            for idx in frame_indices:
                fig = plt.figure(figsize=(6, 3.5))  # adjust aspect to your 30x52 grid if you like
                ax = fig.add_subplot(111)
                ax.imshow(frames[idx], cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.axis('off')
                plt.tight_layout(pad=0)
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
        print(f"Saved multi-page PDF: {pdf_path}")
    else:
        out_paths = []
        for idx in frame_indices:
            fig = plt.figure(figsize=(6, 3.5))
            ax = fig.add_subplot(111)
            ax.imshow(frames[idx], cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.axis('off')
            plt.tight_layout(pad=0)
            out_path = os.path.join(save_dir, f"{base_name}_frame{idx+1}.pdf")
            fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            out_paths.append(out_path)
        print("Saved PDFs:\n  " + "\n  ".join(out_paths))


def parse_layer_ids(value):
    """Parse layer IDs from various formats:
    - Comma-separated: "13,15,17"
    - Space-separated: "13 15 17"
    - List format: "[13,15,17]" or "[0,40]"
    - Range format: "0-5" (expands to [0,1,2,3,4,5])
    - Mixed: "0-2,5,10-12" (expands to [0,1,2,5,10,11,12])
    """
    value = value.strip()
    
    # Handle list format [13,15,17] or [0,40]
    if value.startswith('[') and value.endswith(']'):
        # Remove brackets and parse as comma-separated
        value = value[1:-1]
    
    # Handle range format like "0-5" or mixed "0-2,5,10-12"
    layer_ids = []
    parts = re.split(r'[, ]+', value)  # Split by comma or space
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if '-' in part:
            # Handle range format like "0-5"
            try:
                start, end = map(int, part.split('-'))
                layer_ids.extend(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range format '{part}', skipping")
        else:
            # Handle single number
            try:
                layer_ids.append(int(part))
            except ValueError:
                print(f"Warning: Invalid layer ID '{part}', skipping")
    
    return layer_ids

def load_config_file(config_path):
    """Load configuration from a text file"""
    config = {}
    try:
        with open(config_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'step_id':
                        config['step_id'] = int(value)
                    elif key == 'layer_ids':
                        # Support multiple formats: comma-separated, space-separated, or list format
                        layer_ids = parse_layer_ids(value)
                        config['layer_ids'] = layer_ids
                    elif key == 'inst_id':
                        config['inst_id'] = int(value)
                    else:
                        print(f"Warning: Unknown config key '{key}' on line {line_num}")
                else:
                    print(f"Warning: Invalid config line {line_num}: {line}")
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None
    
    return config

def load_attention_data(pid=0, layer_id=13, step_id=40, inst_id=0, layer_ids=None):
    """Load attention data for a specific PID, layer ID(s), step ID, and instance ID
    If layer_ids is provided (list), average across multiple layers
    """
    attn_path = f"data/attn_maps/wan21_lora/pid{pid}_insts['red kayak'].pt"
    
    if not os.path.exists(attn_path):
        # Try to find the correct file
        attn_dir = "data/attn_maps/wan21_lora/cache_test_gpt-4o/"
        files = [f for f in os.listdir(attn_dir) if f.startswith(f"pid{pid}_")]
        if files:
            attn_path = os.path.join(attn_dir, files[0])
            print(f"Using file: {files[0]}")
        else:
            print(f"No attention file found for PID {pid}")
            return None
    
    print(f"Loading attention map: {attn_path}")
    print(f"Using layer ID: {layer_id}, step ID: {step_id}, instance ID: {inst_id}")
    
    try:
        attn_data = torch.load(attn_path, map_location='cpu')
        
        # Check available steps
        available_steps = list(attn_data['timestep'].keys())
        available_steps.sort()
        print(f"Available steps: {available_steps}")
        
        if step_id not in available_steps:
            print(f"Step {step_id} not found. Available steps: {available_steps}")
            print(f"Using first available step: {available_steps[0]}")
            step_id = available_steps[0]
        
        # Check available layers for the selected step
        available_layers = list(attn_data['timestep'][step_id]['layers'].keys())
        available_layers.sort()
        print(f"Available layers for step {step_id}: {available_layers}")
        
        # Determine which layers to use
        if layer_ids is not None:
            # Use multiple layers for averaging
            layers_to_use = []
            for lid in layer_ids:
                if lid in available_layers:
                    layers_to_use.append(lid)
                else:
                    print(f"Warning: Layer {lid} not found in available layers: {available_layers}")
            
            if not layers_to_use:
                print(f"Error: None of the specified layers {layer_ids} are available. Using first available layer.")
                layers_to_use = [available_layers[0]]
        else:
            # Use single layer
            if layer_id not in available_layers:
                print(f"Layer {layer_id} not found. Available layers: {available_layers}")
                print(f"Using first available layer: {available_layers[0]}")
                layer_id = available_layers[0]
            layers_to_use = [layer_id]
        
        print(f"Using layers: {layers_to_use}")
        
        # Check available instances for the selected step and first layer
        first_layer = layers_to_use[0]
        available_insts = list(attn_data['timestep'][step_id]['layers'][first_layer]['insts'].keys())
        available_insts.sort()
        print(f"Available instances for step {step_id}, layer {first_layer}: {available_insts}")
        
        if inst_id not in available_insts:
            print(f"Instance {inst_id} not found. Available instances: {available_insts}")
            print(f"Using first available instance: {available_insts[0]}")
            inst_id = available_insts[0]
        
        # Extract and average data across multiple layers
        layer_data = []
        for lid in layers_to_use:
            frame_data = attn_data['timestep'][step_id]['layers'][lid]['insts'][inst_id]['attn_map']
            layer_data.append(frame_data.detach().cpu().numpy())
            print(f"Loaded data from layer {lid} with shape: {frame_data.shape}")
        
        # Average across layers
        if len(layer_data) == 1:
            averaged_data = layer_data[0]
            print(f"Using single layer data with shape: {averaged_data.shape}")
        else:
            averaged_data = np.mean(layer_data, axis=0)
            print(f"Averaged across {len(layer_data)} layers, final shape: {averaged_data.shape}")
        
        return averaged_data
        
    except Exception as e:
        print(f"Error loading attention data: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_key_frames(frames, title="Key Frame Attention Maps", save_path=None):
    """Visualize key frames: fixed frames 1, 5, 9, 13, 17, 21 (0-indexed: 0, 4, 8, 12, 16, 20)"""
    print(f"Visualizing key frames from {len(frames)} total frames with shape: {frames.shape}")
    
    # Select fixed key frames: 1, 5, 9, 13, 17, 21 (convert to 0-indexed: 0, 4, 8, 12, 16, 20)
    fixed_frame_indices = [0, 4, 8, 12, 16, 20]  # 0-indexed versions of 1, 5, 9, 13, 17, 21
    total_frames = len(frames)
    
    # Filter to only include frames that exist
    key_frame_indices = [idx for idx in fixed_frame_indices if idx < total_frames]
    
    print(f"Selected key frame indices: {key_frame_indices} (frames: {[idx+1 for idx in key_frame_indices]})")
    
    # Create 1x6 grid for key frames (up to 6 fixed frames)
    num_key_frames = len(key_frame_indices)
    fig_width = 24  # Width for 6 frames
    fig_height = 4  # Height for single row
    
    fig, axes = plt.subplots(1, num_key_frames, figsize=(fig_width, fig_height))
    if num_key_frames == 1:
        axes = [axes]  # Make it iterable
    
    # Find global min/max for consistent color scale
    vmin, vmax = np.min(frames), np.max(frames)
    
    for i, frame_idx in enumerate(key_frame_indices):
        ax = axes[i]
        
        # Create heatmap with correct aspect ratio
        im = ax.imshow(frames[frame_idx], cmap='viridis', aspect='equal', 
                      vmin=vmin, vmax=vmax, interpolation='nearest')
        
        # Determine frame label
        frame_label = f'Frame {frame_idx+1}'
        
        ax.set_title(frame_label, fontsize=12, pad=10)
        ax.set_xlabel('Width (52)', fontsize=10)
        ax.set_ylabel('Height (30)', fontsize=10)
        
        # Set the correct axis limits to match 30x52
        ax.set_xlim(-0.5, 51.5)  # 52 pixels: 0 to 51
        ax.set_ylim(29.5, -0.5)  # 30 pixels: 0 to 29, inverted for imshow
        
        # Add grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.suptitle(f'{title} - Key Frames from {total_frames} total frames', fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Key frames heatmap saved to: {save_path}")
    
    plt.show()

def visualize_full_layers(attn_data, step_id, inst_id, title="Full Layers Visualization", save_path=None):
    """Visualize all available layers, one row per layer, 5 key frames per row"""
    print(f"Creating full layers visualization for step {step_id}, instance {inst_id}")
    
    # Get all available layers for the step
    available_layers = list(attn_data['timestep'][step_id]['layers'].keys())
    available_layers.sort()
    print(f"Available layers: {available_layers}")
    
    if not available_layers:
        print("No layers available for visualization")
        return
    
    # Load data for all layers
    layer_data = {}
    for layer_id in available_layers:
        try:
            frame_data = attn_data['timestep'][step_id]['layers'][layer_id]['insts'][inst_id]['attn_map']
            layer_data[layer_id] = frame_data.detach().cpu().numpy()
            print(f"Loaded layer {layer_id} with shape: {frame_data.shape}")
        except Exception as e:
            print(f"Warning: Could not load layer {layer_id}: {e}")
            continue
    
    if not layer_data:
        print("No valid layer data found")
        return
    
    # Create figure with one row per layer
    num_layers = len(layer_data)
    fig_width = 30  # Width for 6 frames per row
    fig_height = 4 * num_layers  # Height scales with number of layers
    
    fig, axes = plt.subplots(num_layers, 6, figsize=(fig_width, fig_height))
    if num_layers == 1:
        axes = axes.reshape(1, -1)  # Make it 2D
    
    # Find global min/max for consistent color scale across all layers
    all_data = np.concatenate([data.flatten() for data in layer_data.values()])
    vmin, vmax = np.min(all_data), np.max(all_data)
    print(f"Global color scale: min={vmin:.6f}, max={vmax:.6f}")
    
    # Plot each layer
    for row_idx, (layer_id, frames) in enumerate(layer_data.items()):
        # Select fixed key frames: 1, 5, 9, 13, 17, 21 (convert to 0-indexed: 0, 4, 8, 12, 16, 20)
        fixed_frame_indices = [0, 4, 8, 12, 16, 20]  # 0-indexed versions of 1, 5, 9, 13, 17, 21
        total_frames = len(frames)
        
        # Filter to only include frames that exist
        key_frame_indices = [idx for idx in fixed_frame_indices if idx < total_frames]
        
        # Plot up to 6 key frames for this layer
        for col_idx in range(6):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(key_frame_indices):
                frame_idx = key_frame_indices[col_idx]
                frame_data = frames[frame_idx]
                
                # Create heatmap
                im = ax.imshow(frame_data, cmap='viridis', aspect='equal', 
                              vmin=vmin, vmax=vmax, interpolation='nearest')
                
                # Determine frame label
                frame_label = f'Frame {frame_idx+1}'
                
                ax.set_title(f'Layer {layer_id}: {frame_label}', fontsize=10, pad=5)
            else:
                # Empty subplot if we don't have enough frames
                ax.set_visible(False)
            
            # Set axis properties
            ax.set_xlabel('Width (52)', fontsize=8)
            ax.set_ylabel('Height (30)', fontsize=8)
            ax.set_xlim(-0.5, 51.5)
            ax.set_ylim(29.5, -0.5)
            ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.suptitle(f'{title} - Step {step_id}, Instance {inst_id} ({num_layers} layers)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Full layers visualization saved to: {save_path}")
    
    plt.show()

# Animation function removed - only heatmap grid visualization is needed

def show_statistics(frames):
    """Display statistics about the attention maps"""
    print(f"\nAttention Map Statistics:")
    print(f"Shape: {frames.shape}")
    print(f"Min value: {np.min(frames):.6f}")
    print(f"Max value: {np.max(frames):.6f}")
    print(f"Mean value: {np.mean(frames):.6f}")
    print(f"Std value: {np.std(frames):.6f}")
    
    # Per-frame statistics
    print(f"\nPer-frame statistics:")
    for i in range(21):
        frame = frames[i]
        print(f"Frame {i+1:2d}: min={np.min(frame):.4f}, max={np.max(frame):.4f}, mean={np.mean(frame):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Key frames visualization of attention maps")
    parser.add_argument("--pid", type=int, default=0, help="PID to visualize (default: 0)")
    parser.add_argument("--layer_id", type=int, default=13, help="Layer ID to visualize (default: 13)")
    parser.add_argument("--step_id", type=int, default=40, help="Step ID to visualize (default: 40)")
    parser.add_argument("--inst_id", type=int, default=0, help="Instance ID to visualize (default: 0)")
    parser.add_argument("--config", type=str, help="Path to configuration file (overrides other parameters)")
    parser.add_argument("--full", action="store_true", help="Generate full visualization with all layers (one row per layer)")
    parser.add_argument("--save", action="store_true", help="Save visualizations to files")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    parser.add_argument("--frames", type=str,
        help="Selected frames to export (e.g., '0,4,8' or '1-5' or '0,3-5,10'). "
         "Indices are accepted as 0-based or 1-based; 1-based will be converted.")
    parser.add_argument("--single-pdf", action="store_true",
        help="Export selected frames to a single multi-page PDF (used with --frames and --save).")
    parser.add_argument("--no-shared-cmap", action="store_true",
        help="Do not share color scale across selected frames (each page auto-scales).")
    args = parser.parse_args()
    

    # Load configuration from file if provided
    config = None
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config_file(args.config)
        if config is None:
            return
        print(f"Configuration loaded: {config}")
    
    # Determine parameters to use
    if config:
        step_id = config.get('step_id', args.step_id)
        layer_ids = config.get('layer_ids', None)
        inst_id = config.get('inst_id', args.inst_id)
        layer_id = args.layer_id  # Keep original for single layer mode
    else:
        step_id = args.step_id
        layer_ids = None
        inst_id = args.inst_id
        layer_id = args.layer_id
    
    # Display parameters
    if layer_ids:
        print(f"Visualizing attention maps for PID {args.pid}, Step {step_id}, Layers {layer_ids}, Instance {inst_id}")
    else:
        print(f"Visualizing attention maps for PID {args.pid}, Step {step_id}, Layer {layer_id}, Instance {inst_id}")
    print("=" * 80)
    
    # Handle full visualization mode
    if args.full:
        # Load the raw attention data for full visualization
        attn_path = f"data/attn_maps/wan2.1-t2v-14b/pid{args.pid}_insts['red kayak'].pt"
        
        if not os.path.exists(attn_path):
            # Try to find the correct file
            attn_dir = "data/attn_maps/wan2.1-t2v-14b/"
            attn_files = [f for f in os.listdir(attn_dir) if f.startswith(f"pid{args.pid}_") and f.endswith('.pt')]
            if attn_files:
                attn_path = os.path.join(attn_dir, attn_files[0])
                print(f"Found attention file: {attn_path}")
            else:
                print(f"No attention file found for PID {args.pid}")
                return
        
        try:
            attn_data = torch.load(attn_path, map_location='cpu')
            print(f"Loaded attention data from: {attn_path}")
            
            # Check available steps
            available_steps = list(attn_data['timestep'].keys())
            available_steps.sort()
            print(f"Available steps: {available_steps}")
            
            if step_id not in available_steps:
                print(f"Step {step_id} not found. Available steps: {available_steps}")
                print(f"Using first available step: {available_steps[0]}")
                step_id = available_steps[0]
            
            # Create output directory if saving
            if args.save:
                output_dir = f"data/attention_visualizations"
                os.makedirs(output_dir, exist_ok=True)
                full_path = os.path.join(output_dir, f"pid{args.pid}_step{step_id}_full_layers_inst{inst_id}_heatmap.png")
            else:
                full_path = None
            
            # Generate full layers visualization
            title = f"PID {args.pid} - Full Layers Visualization"
            visualize_full_layers(attn_data, step_id, inst_id, title, full_path)
            
            if args.save:
                print(f"\nFull layers heatmap saved to: {output_dir}")
                
        except Exception as e:
            print(f"Error loading attention data for full visualization: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # Regular single/multi-layer visualization
        frames = load_attention_data(args.pid, layer_id, step_id, inst_id, layer_ids)
        # threshold = 0.048
        # mask = frames < threshold
        # frames[mask] = np.random.uniform(0.002, 0.015, size=mask.sum())
        if frames is None:
            return
        
        # Show statistics if requested
        if args.stats:
            show_statistics(frames)
        
        if args.frames:
            # Parse indices after we know total frame count
            total_frames = len(frames)
            frame_indices = parse_frame_indices(args.frames, total_frames=total_frames)
            if not frame_indices:
                print(f"No valid frame indices from '{args.frames}'. Nothing to export.")
                return

            if args.save:
                # Build a base filename reflecting pid/step/layer(s)/inst
                if layer_ids:
                    layer_str = "_".join(map(str, layer_ids))
                    base_name = f"pid{args.pid}_step{step_id}_layers{layer_str}_inst{inst_id}_frames_{args.frames}"
                else:
                    base_name = f"pid{args.pid}_step{step_id}_layer{layer_id}_inst{inst_id}_frames_{args.frames}"

                output_dir = "data/attention_visualizations"
                export_frames_as_pdf(
                    frames=frames,
                    frame_indices=frame_indices,
                    save_dir=output_dir,
                    base_name=base_name if args.single_pdf else base_name,  # suffix handled inside
                    single_pdf=args.single_pdf,
                    cmap='viridis',
                    share_cmap=not args.no_shared_cmap
                )
            else:
                # Show on screen (no axes), one figure per requested frame
                # (useful for quick checks without saving)
                if not frame_indices:
                    print("No frames to display.")
                    return
                data = np.stack([frames[i] for i in frame_indices], axis=0)
                vmin, vmax = (float(np.min(data)), float(np.max(data))) if not args.no_shared_cmap else (None, None)
                for idx in frame_indices:
                    plt.figure(figsize=(6, 3.5))
                    ax = plt.gca()
                    ax.imshow(frames[idx], cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax, interpolation='nearest')
                    ax.axis('off')
                    plt.tight_layout(pad=0)
                    plt.show()

            # When --frames is used, we typically skip the key-frame mosaic to avoid extra plots.
            return
        # Create output directory if saving
        if args.save:
            output_dir = f"data/attention_visualizations"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename based on layer configuration
            if layer_ids:
                layer_str = "_".join(map(str, layer_ids))
                filename = f"pid{args.pid}_step{step_id}_layers{layer_str}_inst{inst_id}_keyframes_heatmap.png"
            else:
                filename = f"pid{args.pid}_step{step_id}_layer{layer_id}_inst{inst_id}_keyframes_heatmap.png"
            
            heatmap_path = os.path.join(output_dir, filename)
        else:
            heatmap_path = None
        
        # Generate title based on layer configuration
        if layer_ids:
            title = f"PID {args.pid}, Step {step_id}, Layers {layer_ids}, Instance {inst_id} - Key Frame Attention Maps"
        else:
            title = f"PID {args.pid}, Step {step_id}, Layer {layer_id}, Instance {inst_id} - Key Frame Attention Maps"
        
        # Visualize key frames
        print("\nCreating key frames visualization...")
        visualize_key_frames(frames, title, heatmap_path)
        
        if args.save:
            print(f"\nHeatmap saved to: {output_dir}")

if __name__ == "__main__":
    main()
