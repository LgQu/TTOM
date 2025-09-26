import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ttom.utils.cache import get_cache, update_cache, cache_init, set_cache_path, get_cache_path
from ttom.utils.prompt_convert import enrich_prompt
from ttom.utils.object_extracter import extract_object_metadata_with_llm, verify_object_metadata_with_llm
from ttom.utils.layout_handler import generate_layout_with_llm, extract_layout_from_response, generate_gif_from_layout, interpolate_layout_boxes
import ast
import json
import argparse


def create_prompt_cache_from_txt(file_path, json_path):
    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    file_name = file_path.split("/")[-1].split(".")[0]
    cache_dict = {}
    for idx, prompt in enumerate(prompts):
        cache_dict[str(idx)] = {
            "original_prompt": prompt
        }
    with open(json_path, "w") as json_file:
        json.dump(cache_dict, json_file, indent=4)
    print(f"✅ Prompt cache written to: {json_path}")
    return len(prompts)


def enrich_prompt_and_get_object_metadata(prompt=None, template_type="default", prompt_id=None, skip_if_exists=False):

    cache_init()
    cache = get_cache()
    # 1. prepare prompt

    if prompt_id is not None:
        print(f"\nprompt_id: {prompt_id}")
        # prompt = get_prompt(prompt_id)
        prompt = cache[prompt_id]["original_prompt"]
        print(f"\nprompt: {prompt}")
    elif prompt is not None:
        prompt = prompt
        print(f"\nprompt: {prompt}")
    else:
        raise ValueError("prompt or prompt_id must be provided")

    # 2. enrich prompt
    if skip_if_exists and "enriched_prompt" in cache[prompt_id] and cache[prompt_id]["enriched_prompt"]:
        print(f"[⚡ Skip] Object metadata already exists for {prompt_id}")
        prompt_converted = cache[prompt_id]["enriched_prompt"]
    else:
        prompt_converted = enrich_prompt(prompt, prompt_id=prompt_id, template_type=template_type)
        print(f"\nEnriched prompt: {prompt_converted}")
        cache[prompt_id]["enriched_prompt"] = prompt_converted

    # 3. extract object metadata
    if skip_if_exists and "objects" in cache[prompt_id] and cache[prompt_id]["objects"]:
        print(f"[⚡ Skip] Object metadata already exists for {prompt_id}")
        return
    else:
        object_metadata = extract_object_metadata_with_llm(prompt, prompt_converted)
    # print(f"\nObject metadata: {object_metadata}")

    # 4. verify object metadata
    verified_object_metadata = verify_object_metadata_with_llm(prompt, prompt_converted, object_metadata)
    # print(f"\nVerified object metadata: {verified_object_metadata}")

    # 5. parse result
    parsed_result = ast.literal_eval(verified_object_metadata)
    print(f"\nParsed result: {parsed_result}")

    # 6. update cache
    cache[prompt_id]["objects"] = {}
    for obj_id, obj_data in parsed_result["objects"].items():
        cache[prompt_id]["objects"][obj_id] = {}
        cache[prompt_id]["objects"][obj_id]["object_id"] = obj_data["object_id"]
        cache[prompt_id]["objects"][obj_id]["object"] = obj_data["object"]
        cache[prompt_id]["objects"][obj_id]["object_phrases"] = obj_data["object_phrases"]
        cache[prompt_id]["objects"][obj_id]["object_words"] = obj_data["object_words"]
        cache[prompt_id]["objects"][obj_id]["object_positions"] = []

    update_cache(cache)

def generate_first_frame_prompt(prompt=None, prompt_id=None, skip_if_exists=False):
    cache_init()
    cache = get_cache()

    # 1. Get enriched_prompt and layout
    if prompt_id is not None:
        print(f"\n[prompt_id]: {prompt_id}")
        prompt_entry = cache.get(prompt_id, {})
        prompt = prompt_entry.get("enriched_prompt")
        layout = prompt_entry.get("layout_with_reasoning")

        if not prompt:
            raise ValueError(f"[❌ ERROR] No enriched_prompt found for prompt_id: {prompt_id}")
        print(f"\n[Input Enriched Prompt]: {prompt}")
        if not layout:
            print(f"[⚠️ WARNING] No layout_with_reasoning found for prompt_id: {prompt_id}")
    elif prompt is not None:
        layout = None
        print(f"\n[Input Prompt]: {prompt}")
    else:
        raise ValueError("Either prompt or prompt_id must be provided.")

    # 2. Whether to skip existing
    if skip_if_exists and "first_frame_prompt" in cache.get(prompt_id, {}):
        print(f"[⚡ Skip] First-frame prompt already exists for {prompt_id}")
        return

    # 3. Generate first frame visual description
    first_frame_prompt = extract_first_frame_prompt(prompt, layout)
    print(f"\n[✅ Generated First Frame Prompt]: {first_frame_prompt}")

    # 4. Update cache
    cache[prompt_id]["first_frame_prompt"] = first_frame_prompt
    update_cache(cache)


def generate_layout(prompt_id, skip_if_exists=False):
    cache_init()
    cache = get_cache()

    if skip_if_exists and "layout" in cache[prompt_id] and "frames" in cache[prompt_id]["layout"]:
        print(f"[⚡ Skip] Layout already exists for {prompt_id}")
        return

    raw_metadata = cache[prompt_id]
    # print(f"\nRaw metadata: {raw_metadata}")
    metadata = {
        "original_prompt": raw_metadata["original_prompt"],
        "enriched_prompt": raw_metadata["enriched_prompt"],
        "objects": raw_metadata["objects"]
    }
    print(f"\nMetadata: {metadata}")
    layout_with_reasoning = generate_layout_with_llm(metadata)
    print(f"\nLayout with reasoning: {layout_with_reasoning}")
    cache[prompt_id]["layout_with_reasoning"] = layout_with_reasoning
    layout = extract_layout_from_response(layout_with_reasoning)
    print(f"\nLayout: {layout}")
    cache[prompt_id]["layout"] = layout

    update_cache(cache)

def prune_unused_objects_and_reindex(prompt_id):
    cache_init()
    cache = get_cache()

    layout = cache[prompt_id]["layout"]
    objects = cache[prompt_id]["objects"]

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
            obj["id"] = name_to_new_id[obj["name"]]

    cache[prompt_id]["objects"] = new_objects
    cache[prompt_id]["layout"] = layout

    update_cache(cache)
    print(f"✅ Pruned and reindexed objects for prompt_id: {prompt_id}")

def show_layout(prompt_id, json_path, skip_if_exists=False):
    cache_init()
    cache = get_cache()
    layout = cache[prompt_id]["layout"]

    box_dir = f"data/layout/boxes_{json_path.split('/')[-1].split('.')[0]}"
    os.makedirs(box_dir, exist_ok=True)
    path = f"{box_dir}/layout_{prompt_id}.gif"
    if skip_if_exists and os.path.exists(path):
        print(f"[⚡ Skip] Layout GIF already exists for {prompt_id}")
        return
    generate_gif_from_layout(layout, save_path=path, fps=2)

def interpolate_layout(prompt_id):
    cache_init()
    cache = get_cache()
    layout = cache[prompt_id]["layout"]
    interpolated_layout = interpolate_layout_boxes(layout)
    print(f"\nInterpolated layout: {interpolated_layout}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process prompts without enrichment')
    parser.add_argument('--benchmark_source', "-bs", type=str, choices=["t2vcompbench", "vbench"], required=True,
                        help='Benchmark source: t2vcompbench or vbench')
    parser.add_argument('--benchmark_type', "-bt", type=str, required=True,
                        help='Benchmark type: for t2vcompbench: 1_consistent_attr, 2_dynamic_attr, 3_spatial_relationship, 4_motion_binding, 5_action_binding, 6_interaction, 7_numeracy; for vbench: appearance_style, color, human_action, multiple_objects, overall_consistency, scene, spatial_relationship, subject_consistency, temporal_flickering, temporal_style')
    parser.add_argument('--start_idx', type=int, default=1, help='Start index for processing (default: 0)')
    parser.add_argument('--end_idx', type=int, default=201, help='End index for processing (default: 200)')
    parser.add_argument('--skip_if_exists', action='store_true', help='Skip processing if results already exist')
    parser.add_argument('--create_cache', action='store_true', help='Create cache from txt file first')
    args = parser.parse_args()
    print(f"Using benchmark source: {benchmark_source}")
    print(f"Using benchmark type: {benchmark_type}")


    txt_path = f"/hpctmp/e1351271/lvd/LVD_extention/cache/{benchmark_source}_prompts/{benchmark_type}.txt"
    json_path = f"cache/{benchmark_source}_{benchmark_type}-gpt_4o.json"
    set_cache_path(json_path)
    # count = create_prompt_cache_from_txt(txt_path, json_path)
    # print(f"✅ Created prompt cache: {json_path}")
    # print(f"✅ Count: {count}")
    count = 1

    # 1. enrich prompt and get object metadata
    for i in range(0, count):
        # enrich_prompt_and_get_object_metadata(prompt_id = str(i), template_type="default", skip_if_exists=True)
        # generate_layout(prompt_id = str(i), skip_if_exists=True)
        # generate_first_frame_prompt(prompt_id=str(i), skip_if_exists=True)
        # prune_unused_objects_and_reindex(prompt_id=str(i))
        show_layout(prompt_id = str(i), json_path=json_path, skip_if_exists=False)
    # pid_list = [90,100,101,112,119]
    # pid_list = [i for i in range(200)]
    # for i in pid_list:
        # enrich_prompt_and_get_object_metadata(prompt_id = str(i), skip_if_exists=False)
        # generate_layout(prompt_id = str(i), skip_if_exists=False)
        # prune_unused_objects_and_reindex(prompt_id=str(i))
        # show_layout(prompt_id = str(i), skip_if_exists=False)
    # enrich_prompt_and_get_object_metadata(prompt_id = "91", pass_enriched_prompt=True)

    # # 2. generate layout
    # for i in range(90, 200):

    # show_layout(prompt_id = "1")
    # interpolate_layout(prompt_id = "1")