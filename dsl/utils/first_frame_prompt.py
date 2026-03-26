import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsl.utils.cache import get_cache, update_cache, cache_init

def generate_first_frame_prompt(prompt_id):
    cache = get_cache()
    prompt = cache[prompt_id]["prompt"]
    return prompt


