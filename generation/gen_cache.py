"""
Phase 1: Meta extraction and layout generation for CogVideoX.

Uses GPT-4o to:
1. Enrich prompts with detailed descriptions
2. Extract object metadata (entities, phrases, positions)
3. Generate spatial-temporal bounding box layouts

Usage:
    python generation/gen_cache.py \
        --benchmark_source t2vcompbench \
        --benchmark_type 4_motion_binding \
        --start_idx 0 \
        --end_idx 199 \
        --skip_if_exists
"""
import sys
import os
import argparse
import json
import ast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dsl.utils.cache import get_cache, update_cache, cache_init, set_cache_path
from dsl.utils.prompt_convert import enrich_prompt
from dsl.utils.object_extracter import extract_object_metadata_with_llm, verify_object_metadata_with_llm
from dsl.utils.layout_handler import (
    generate_layout_with_llm,
    extract_layout_from_response,
    generate_gif_from_layout,
    interpolate_layout_boxes,
)


def create_prompt_cache_from_txt(file_path, output_path):
    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    cache_dict = {}
    for idx, prompt in enumerate(prompts):
        cache_dict[str(idx)] = {"original_prompt": prompt}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache_dict, f, indent=4)
    print(f"Prompt cache written to: {output_path}")
    return output_path


def enrich_and_extract_metadata(prompt_id, skip_if_exists=False):
    cache_init()
    cache = get_cache()

    entry = cache.get(prompt_id)
    if entry is None:
        raise ValueError(f"prompt_id '{prompt_id}' not found in cache")

    prompt = entry["original_prompt"]
    print(f"\n[{prompt_id}] prompt: {prompt}")

    if skip_if_exists and entry.get("enriched_prompt") and entry.get("objects"):
        print(f"[Skip] Metadata already exists for {prompt_id}")
        return

    enriched = enrich_prompt(prompt, prompt_id=prompt_id, skip_verify=skip_if_exists)
    print(f"[{prompt_id}] enriched: {enriched}")
    cache[prompt_id]["enriched_prompt"] = enriched

    raw_metadata = extract_object_metadata_with_llm(enriched)
    verified_metadata = verify_object_metadata_with_llm(enriched, raw_metadata)
    parsed = ast.literal_eval(verified_metadata)
    print(f"[{prompt_id}] objects: {parsed}")

    cache[prompt_id]["objects"] = {}
    for obj_id, obj_data in parsed["objects"].items():
        cache[prompt_id]["objects"][obj_id] = {
            "object_id": obj_data["object_id"],
            "object": obj_data["object"],
            "object_phrases": obj_data["object_phrases"],
            "object_words": obj_data["object_words"],
            "object_positions": [],
        }

    update_cache(cache)


def generate_layout(prompt_id, skip_if_exists=False):
    cache_init()
    cache = get_cache()

    if skip_if_exists and "layout" in cache[prompt_id] and "frames" in cache[prompt_id]["layout"]:
        print(f"[Skip] Layout already exists for {prompt_id}")
        return

    entry = cache[prompt_id]
    metadata = {
        "original_prompt": entry["original_prompt"],
        "enriched_prompt": entry["enriched_prompt"],
        "objects": entry["objects"],
    }
    print(f"\n[{prompt_id}] Generating layout...")
    layout_response = generate_layout_with_llm(metadata)
    cache[prompt_id]["layout_with_reasoning"] = layout_response

    layout = extract_layout_from_response(layout_response)
    print(f"[{prompt_id}] Layout: {layout}")
    cache[prompt_id]["layout"] = layout

    update_cache(cache)


def prune_unused_objects(prompt_id):
    cache_init()
    cache = get_cache()

    layout = cache[prompt_id]["layout"]
    used_names = set()
    for frame_objs in layout["frames"].values():
        for obj in frame_objs:
            used_names.add(obj["name"])

    new_objects = {}
    name_to_new_id = {}
    new_id = 0
    for old_id, obj in sorted(cache[prompt_id]["objects"].items(), key=lambda x: int(x[0])):
        if obj["object"] in used_names:
            name_to_new_id[obj["object"]] = new_id
            obj["object_id"] = str(new_id)
            new_objects[str(new_id)] = obj
            new_id += 1

    for frame_id, frame_objs in layout["frames"].items():
        for obj in frame_objs:
            obj["id"] = name_to_new_id.get(obj["name"], obj["id"])

    cache[prompt_id]["objects"] = new_objects
    cache[prompt_id]["layout"] = layout
    update_cache(cache)
    print(f"Pruned objects for prompt_id: {prompt_id}")


def save_layout_gif(prompt_id, output_dir, skip_if_exists=False):
    cache_init()
    cache = get_cache()
    layout = cache[prompt_id]["layout"]

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"layout_{prompt_id}.gif")
    if skip_if_exists and os.path.exists(path):
        print(f"[Skip] Layout GIF already exists: {path}")
        return
    generate_gif_from_layout(layout, save_path=path, fps=2)
    print(f"Layout GIF saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Meta extraction and layout generation")
    parser.add_argument("--benchmark_source", type=str, default="t2vcompbench")
    parser.add_argument("--benchmark_type", type=str, default="4_motion_binding")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=199)
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--txt_path", type=str, default=None, help="Path to prompt .txt file to initialize cache")
    parser.add_argument("--steps", type=str, default="all",
                        help="Comma-separated steps to run: enrich,layout,prune,gif,all")
    args = parser.parse_args()

    cache_name = f"cache_{args.benchmark_type}_gpt-4o"
    cache_path = f"cache/{cache_name}.json"
    set_cache_path(cache_path)

    if args.txt_path:
        create_prompt_cache_from_txt(args.txt_path, cache_path)

    steps = args.steps.split(",") if args.steps != "all" else ["enrich", "layout", "prune", "gif"]
    gif_dir = f"data/layout/boxes_{cache_name}"

    for pid in range(args.start_idx, args.end_idx + 1):
        pid_str = str(pid)
        try:
            if "enrich" in steps:
                enrich_and_extract_metadata(pid_str, skip_if_exists=args.skip_if_exists)
            if "layout" in steps:
                generate_layout(pid_str, skip_if_exists=args.skip_if_exists)
            if "prune" in steps:
                prune_unused_objects(pid_str)
            if "gif" in steps:
                save_layout_gif(pid_str, gif_dir, skip_if_exists=args.skip_if_exists)
        except Exception as e:
            print(f"[Error] prompt_id={pid_str}: {e}")
            continue

    print(f"\nDone. Cache: {cache_path}, Layout GIFs: {gif_dir}/")


if __name__ == "__main__":
    main()
