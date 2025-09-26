"""
The CogVideoX model is designed to generate high-quality videos based on detailed and highly descriptive prompts.
The model performs best when provided with refined, granular prompts, which enhance the quality of video generation.
This script is designed to assist with transforming simple user inputs into detailed prompts suitable for CogVideoX.
It can handle both text-to-video (t2v) and image-to-video (i2v) conversions.

- For text-to-video, simply provide the prompt.
- For image-to-video, provide the path to the image file and an optional user input.
The image will be encoded and sent as part of the request to Azure OpenAI.

### How to run:
Run the script for **text-to-video**:
    $ python convert_demo.py --prompt "A girl riding a bike." --type "t2v"

Run the script for **image-to-video**:
    $ python convert_demo.py --prompt "the cat is running" --type "i2v" --image_path "/path/to/your/image.jpg"
"""

import argparse
from openai import OpenAI, AzureOpenAI
import base64
from mimetypes import guess_type
import os
import json
from ttom.utils.cache import get_cache, update_cache, cache_init, get_prompt_id_from_cache
from ttom.utils.api_key import api_key

sys_prompt_t2v = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.

*Always preserve the original prompt’s core content. For example, if the original prompt describes a motion such as "to the left side", please explicitly include both the starting location and the destination. For instance, revise the description to say "from the right side to the left side". Integrate all given details faithfully into your expanded description.

user input:
"""

sys_prompt_i2v = """
**Objective**: **Give a highly descriptive video caption based on input image and user input. **. As an expert, delve deep into the image with a discerning eye, leveraging rich creativity, meticulous thought. When describing the details of an image, include appropriate dynamic information to ensure that the video caption contains reasonable actions and plots. If user input is not empty, then the caption should be expanded according to the user's input. 

**Note**: The input image is the first frame of the video, and the output video caption should describe the motion starting from the current image. User input is optional and can be empty. 

**Note**: Don't contain camera transitions!!! Don't contain screen switching!!! Don't contain perspective shifts !!!

**Answering Style**:
Answers should be comprehensive, conversational, and use complete sentences. The answer should be in English no matter what the user's input is. Provide context where necessary and maintain a certain tone.  Begin directly without introductory phrases like "The image/video showcases" "The photo captures" and more. For example, say "A woman is on a beach", instead of "A woman is depicted in the image".

**Output Format**: "[highly descriptive image caption here]"

user input:
"""

sys_prompt_numeracy = """
You are an expert scene analyst specialized in identifying and counting objects in visual descriptions.
Your task is to carefully analyze the user's scene description and enumerate all distinct objects present.

Rules to follow:

- Always count each object instance individually; do not group multiple similar objects into one.
- Provide a clear and concise description of each object and its attributes.
- Preserve the original meaning and content of the user's description without omitting details.
- Your output should be a natural language explanation stating the total number of objects and their types.

Answer only based on the textual description provided; do not assume anything beyond it.

user input:
"""


def image_to_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def build_prompt(text: str, template_type: str = "default") -> list:
    if template_type == "default":
        return build_default_prompt(text)
    elif template_type == "numeracy":
        return build_numeracy_prompt(text)
    else:
        raise ValueError(f"Unknown template type: {template_type}")


def build_default_prompt(text: str, sys_prompt: str = sys_prompt_t2v) -> list:
    return [
            {"role": "system", "content": f"{sys_prompt}"},
            {
                "role": "user",
                "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " a girl is on the beach"',
            },
            {
                "role": "assistant",
                "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
            },
            {
                "role": "user",
                "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A man jogging on a football field"',
            },
            {
                "role": "assistant",
                "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
            },
            {
                "role": "user",
                "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "Three robots dance rhythmically in the lab."',
            },
            {
                "role": "assistant",
                "content": "In a futuristic laboratory filled with glowing screens and intricate machinery, three humanoid robots engage in a synchronized dance. Robot 1, a sleek silver figure, leads with sharp, precise movements, its joints flexing effortlessly under the ambient blue and white lights. Robot 2, adorned with subtle chrome-blue accents, follows closely, its motions slightly more fluid, weaving graceful arcs through the air. Robot 3, distinguished by a polished black finish, adds dynamic spins and agile footwork, injecting bursts of energy into the routine. The lab's high-tech environment, with blinking consoles and holographic displays, serves as a vivid backdrop. As they perform, each robot’s LED eyes flash in perfect rhythm with the upbeat electronic music, amplifying the vibrancy of their captivating performance.",
            },
            {
                "role": "user",
                "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A cat is slinking to the left side of a room"',
            },
            {
                "role": "assistant",
                "content": "A sleek gray tabby cat with vivid green eyes slinks quietly toward the left side of a cozy living room. Gentle firelight from a nearby fireplace flickers across the walls, where family photos hang in soft shadow. The cat moves low past a burgundy armchair and a wooden coffee table holding an open book and a steaming mug of tea. A thick woven rug muffles its steps as it glides gracefully through the warm, softly lit space, its fur glowing subtly in the ambient light.",
            },
            {
                "role": "user",
                "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: " {text} "',
            },
        ]

def build_numeracy_prompt(text: str, sys_prompt: str = sys_prompt_numeracy) -> list:
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": 'Three chairs sit around a campfire, and seven people roasting marshmallows.',
        },
        {
            "role": "assistant",
            "content": "chair_1, chair_2, and chair_3 are arranged around a glowing campfire, their wooden frames casting soft shadows on the ground. Nearby, person_1, person_2, person_3, person_4, person_5, person_6, and person_7 are gathered closely, roasting marshmallows over flickering flames, chatting and laughing as the night sky deepens. The warm light illuminates their faces, and a gentle breeze rustles the nearby trees, creating an intimate, cozy atmosphere."
        },
        {
            "role": "user",
            "content": "Four dogs play in the sunny park, while two children watch nearby."
        },
        {
            "role": "assistant",
            "content": (
                "dog_1, dog_2, dog_3, and dog_4 frolic energetically across the lush green grass of the sunny park, "
                "their tails wagging joyfully as they chase after one another. Not far off, child_1 and child_2 sit on a wooden bench, "
                "their eyes bright with amusement and smiles spreading across their faces. The scene is filled with laughter, "
                "the sounds of playful barks, and the gentle rustling of leaves in the warm breeze."
            )
        },
        {
            "role": "user",
            "content": "Five red apples hang on the tree, and a bluebird perches on a branch."
        },
        {
            "role": "assistant",
            "content": (
                "apple_1, apple_2, apple_3, apple_4, and apple_5 hang heavily on the branches of the tree, their shiny red skins catching "
                "the sunlight. Nearby, bluebird_1 perches quietly on a slender branch, its feathers gleaming in the morning light. "
                "The gentle rustling of leaves accompanies the peaceful atmosphere of the orchard."
            )
        },
        {
            "role": "user",
            "content": f'{text}',
        },
    ]


def enrich_prompt(prompt: str, prompt_id: str = None, template_type: str = "default", retry_times: int = 1, type: str = "t2v", image_path: str = None, skip_verify: bool = False):
    """
    Convert a prompt to a format that can be used by the model for inference
    """
    cache_init()
    cache = get_cache()
    record_id = get_prompt_id_from_cache(prompt)

    if record_id is not None and cache[record_id].get("enriched_prompt") is not None and skip_verify:
        print(f"\nPrompt found in cache under id: {record_id}")
        return cache[record_id].get("enriched_prompt")
    
    client = OpenAI(api_key=api_key)

    text = prompt.strip()
    enriched_prompt = None
    for i in range(retry_times):
        if type == "t2v":
            messages = build_prompt(text, template_type)
            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4o",  # glm-4-plus and gpt-4o have be tested
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=250,
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"{sys_prompt_i2v}"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_to_url(image_path),
                                },
                            },
                        ],
                    },
                ],
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=250,
            )
        if response.choices:
            enriched_prompt = response.choices[0].message.content
            if prompt_id is not None:
                new_id = prompt_id
            else:
                used_ids = set(int(k) for k in cache.keys())
                new_id = 0
                while new_id in used_ids:
                    new_id += 1
                
            cache[new_id] = {
                "original_prompt": prompt,
                "enriched_prompt": enriched_prompt,
                "objects": {}
            }
            update_cache(cache)

    return enriched_prompt
