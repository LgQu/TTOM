import json

# cache_file = "cache/cache_demo_gpt-4o_converted.json"
# cache_file = "cache/cache_3_spatial_relationship_gpt-4o.json"
cache_file = "cache/cache_4_motion_binding_gpt-4o.json"
cache = {}

def cache_init():
    global cache
    with open(cache_file, "r", encoding="utf-8") as f:
        cache = json.load(f)

def get_prompt_from_cache(prompt_id):
    return cache[prompt_id]["original_prompt"]

def get_cache():
    return cache

def update_cache(cache_data):
    global cache
    cache = cache_data
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Cache updated and saved to {cache_file}")

def get_prompt_id_from_cache(prompt):
    for k, v in cache.items():
        if v["original_prompt"] == prompt:
            return k
    return None

def get_objects_from_cache(prompt_id: str):
    if prompt_id in cache:
        return cache[prompt_id].get("objects")
    return None

def get_cache_path():
    return cache_file

def set_cache_path(cache_path):
    global cache_file
    cache_file = cache_path