from openai import OpenAI
import json
from ttom.utils.api_key import api_key

client = OpenAI(api_key=api_key)


templatev0_1_object_metadata = [
    {
        "role": "system",
        "content": """You are an intelligent assistant that extracts **object metadata** from descriptive paragraphs.

You will receive a descriptive paragraph. Your job is to:
1. Identify all distinct **objects** in the paragraph. Objects must be **concrete physical entities** that play a **visible or active role** in the paragraph.
2. Do **not** extract parts, components, or attributes of major objects (e.g. "wings", "tires", "eyes") as separate objects.
3. You will also receive an original_prompt. Only extract objects that literally appear in the original_prompt (case-insensitive).
4. For each object:
   - Assign a unique "object_id".
   - Provide the "object" canonical name (e.g. "woman", "kite").
   - Under "object_phrases", list all **quoted phrases** from the paragraph that explicitly refer to the object, each phrase at least 3 words. Preserve the original case and formatting exactly as it appears in the paragraph.
   - Under "object_words", list all words that are used to refer to the object in "object_phrases". Each item in this list must be a single word, and must strictly correspond to one of the "object_phrases" entries. Preserve the original case and formatting exactly as it appears in the paragraph.
5. If multiple objects with the same name appear, keep the name the same, but assign different ids.

Respond in the following format:

{
  "objects": {
    "0<unique identifier for object>": {
      "object_id": "<unique identifier for object>",
      "object": "<canonical name of object>",
      "object_phrases": [
          "<quoted phrase 1>",
          "<quoted phrase 2>",
          ...
      ],
      "object_words": [
          "<word 1>",
          "<word 2>",
          ...
      ]
    }
  ...
}


Guidelines:
- Exclude abstract ideas, emotions, locations, and background elements unless they're the focus of description.
- Do not include synonyms unless they are literally mentioned in the paragraph.
- Preserve the **quoted form** (as in the paragraph).
- **Strictly preserve the original case and formatting** from the paragraph—do not change capitalization or punctuation.
- Under `"object_words"`, list only **single words** that correspond to entries in `"object_phrases"`, matching them exactly in form.
"""
    },
    {
        "role": "user",
        "content": """Description:
original_prompt:
A car glides smoothly along a city street.
Enriched prompt:
A sleek, modern sedan glides smoothly along a bustling city street, its glossy black exterior reflecting the neon glow of nearby billboards. The car moves effortlessly from left to right, its headlights slicing through the twilight air as it passes towering skyscrapers and dimly lit alleyways. Moments later, the vehicle picks up speed, weaving gracefully through light traffic, its tires hugging the asphalt with precision. The cityscape morphs behind it—vibrant storefronts blur into a cascade of color while streetlights flicker rhythmically above."""
    },
    {
        "role": "assistant",
        "content": """
{
  "objects": {
    "0": {
      "object_id": "0",
      "object": "sedan",
      "object_phrases": [
          "modern sedan glides",
          "The car moves",
          "the vehicle picks"
      ],
      "object_words": [
          "sedan",
          "car",
          "vehicle"
      ]
    }
  }
}"""
    },
    {
        "role": "user",
        "content": """Description:
          original_prompt:
          A child plays with a bright yellow kite in an open field.
          Enriched prompt:
          A child plays with a bright yellow kite in an open field. The kite soars high above, dipping and dancing with the wind. The boy laughs as he runs, holding tightly to the string. In the distance, a few birds fly by and clouds drift slowly across the sky."""
    },
    {
        "role": "assistant",
        "content": """
{
  "objects": {
    "0": {
      "object_id": "0",
      "object": "child",
      "object_phrases": ["A child plays with", "The boy laughs"],
      "object_words": ["child", "boy"]
    },
    "1": {
      "object_id": "1",
      "object": "kite",
      "object_phrases": ["a bright yellow kite", "The kite soars"],
      "object_words": ["kite", "kite"]
    }
  }
}"""
    }
]


def extract_object_metadata_with_llm(original_prompt: str, enriched_prompt: str):
    messages = templatev0_1_object_metadata.copy()
    messages.append({"role": "user", "content": "original_prompt:\n" + original_prompt.strip() + "\n\nenriched_prompt:\n" + enriched_prompt.strip()})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content

def verify_object_metadata_with_llm(original_prompt: str, enriched_prompt: str, metadata: str):
    messages = templatev0_1_object_metadata.copy()
    messages.append(
        {"role": "user", 
         "content": """
              Please check if the extracted object metadata strictly follows the guidelines. 
              If there are any issues (e.g. inclusion of background elements, abstract parts, possessive pronouns like "its", or mismatches between `object_phrases` and `object_words`), correct them.
              Return ONLY the corrected `"objects"` field in the exact same format. If there are no problems, return the original unchanged `"objects"` field.
              - **Do not include possessive pronouns** (such as `"its"`, `"his"`, `"her"`, `"their"`) in `object_phrases`, as they typically refer to parts or attributes of the object rather than the whole.
              - **`object_words` must only contain base object words**, without any possessive suffixes like `'s`. For example, use `"car"` instead of `"car's"`, and `"dog"` instead of `"dog's"`.  
              - Make sure that `object_words` are strictly nouns representing whole entities, not their parts or attributes.
              Respond in the following format:

              {
                "objects": {
                  "0<unique identifier for object>": {
                    "object_id": "<unique identifier for object>",
                    "object": "<canonical name of object>",
                    "object_phrases": [
                        "<quoted phrase 1>",
                        "<quoted phrase 2>",
                        ...
                    ],
                    "object_words": [
                        "<word 1>",
                        "<word 2>",
                        ...
                    ]
                  }
                ...
              }
              Description:
          """ 
          + "original_prompt:\n" + original_prompt.strip()
          + "\n\nEnriched prompt:\n" + enriched_prompt.strip()
          + '''
          Object metadata to be verified:
          '''
          + metadata.strip()
          })
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )
    cleaned = strip_json_markdown_block(response.choices[0].message.content)
    return cleaned

def strip_json_markdown_block(text):
    """
    Remove markdown code block markers like ```json ... ```
    """
    if text.strip().startswith("```"):
        lines = text.strip().splitlines()
        # remove first and last line (```json ... ```)
        return "\n".join(line for line in lines[1:] if not line.strip().startswith("```"))
    return text

if __name__ == "__main__":
    paragraph = """
    A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.
    """

    # Run the extractor
    result = extract_object_metadata_with_llm(paragraph)
    print(result)
    import ast

    parsed_result = ast.literal_eval(result)
    print(parsed_result)