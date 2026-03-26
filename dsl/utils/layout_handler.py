import re
import ast
from openai import OpenAI
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from scipy.interpolate import interp1d
import imageio
import os
from PIL import Image, ImageDraw, ImageFont
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

templatev0_1_layout = [
{
  "role": "system",
  "content": '''You are an intelligent bounding box generator for videos. You do not generate videos, only bounding boxes for visual objects described in a video caption.

You will receive three inputs:
1. original_prompt – defines core motion intent, must be strictly followed.
2. converted_prompt – a paragraph describing a 6-frame video scene (2 fps), enriched with visual and contextual details.
3. object metadata – a list of physical entities, aliases, and their exact phrases.

Your task: generate realistic bounding boxes per frame for each main object, plus a background keyword. Include only visually dominant, central, or thematically important objects from the metadata. Ignore minor/background entities.

Bounding box rules:
- Follow motion path and direction from original_prompt exactly. Use converted_prompt only for additional visual cues (e.g., appearance, lighting).
- Resolution: 720×480 (top-left: [0,0], bottom-right: [720,480]).
- Format: {'id': <int>, 'name': <object name>, 'box': [x, y, width, height]}.
- Keep consistent id across frames. Boxes can go out of bounds if physically reasonable.
- Do not invent or add objects not in the metadata. Only use the important objects from the list.
- Match box size to camera framing:
  - Close-ups: large boxes (e.g., ≥500×300).
  - Unless explicitly described as small in the prompt, do not generate boxes smaller than 200×150.
  - Adjust for zoom, pan, or tracking.
- Avoid overlap unless implied (e.g., interaction).

Output format:
- A brief reasoning paragraph (2–3 sentences) explaining object motion and camera behavior.
- Frames 1 to 6: each as a list of bounding box dicts.
- Final line: Background keyword: <word>

No markdown, comments, or formatting.'''
},

    {
        "role": "user",
        "content": '''{
    "original_prompt": "A realistic lively video of a scene with one moving car",
    "enriched_prompt": "A sleek, silver sedan glides smoothly along a bustling city street, its polished exterior reflecting the vibrant urban landscape. The car's headlights illuminate the road ahead as it navigates through the lively scene, passing towering skyscrapers and bustling pedestrians. The camera captures the rhythmic motion of the wheels against the asphalt, while the city lights create a dazzling display of colors on the car's surface. As it moves, the sound of the engine harmonizes with the distant hum of city life, creating an immersive experience. The scene is set against a backdrop of a setting sun, casting a warm glow over the entire cityscape.",
    "objects": {
      "0": {
        "object_id": "0",
        "object": "sedan",
        "object_phrases": [
          "A sleek, silver sedan glides",
          "The car's headlights illuminate",
          "As it moves"
        ],
        "object_words": [
          "sedan",
          "car's",
          "it"
        ],
        "object_positions": []
      }
    }
  }''',
    },
    {
  "role": "assistant",
  "content": '''The sedan moves smoothly from the left side of the frame to the right over the course of the 6 frames, reflecting its steady gliding motion through a city street. Due to perspective and camera tracking, the car's size slightly increases as it approaches the center, then slightly decreases as it exits. Lighting changes and reflections are implied but not modeled here.

Frame 1: [{'id': 0, 'name': 'sedan', 'box': [50, 300, 300, 200]}];
Frame 2: [{'id': 0, 'name': 'sedan', 'box': [120, 300, 300, 200]}];
Frame 3: [{'id': 0, 'name': 'sedan', 'box': [200, 298, 300, 200]}];
Frame 4: [{'id': 0, 'name': 'sedan', 'box': [300, 295, 300, 200]}];
Frame 5: [{'id': 0, 'name': 'sedan', 'box': [420, 292, 300, 200]}];
Frame 6: [{'id': 0, 'name': 'sedan', 'box': [540, 290, 300, 200]}];
Background keyword: city street'''
}

]

def generate_layout_with_llm(prompt_metadata):
    messages = templatev0_1_layout.copy()
    messages.append({"role": "user", "content": json.dumps(prompt_metadata, indent=2)})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )
    layout_from_llm = response.choices[0].message.content
    # print("\nLayout from LLM:", layout_from_llm)
    return layout_from_llm

def extract_layout_from_response(response_text):
    layout = {}
    background = None

    frame_pattern = r"Frame\s+(\d+)\s*:\s*(\[.*?\]);"
    matches = re.findall(frame_pattern, response_text, re.DOTALL)

    for frame_id, bbox_str in matches:
        try:
            boxes = ast.literal_eval(bbox_str.strip())
            layout[int(frame_id)] = boxes
        except Exception as e:
            print(f"Error parsing frame {frame_id}: {e}")

    bg_match = re.search(r"Background keyword\s*:\s*(.+)", response_text)
    if bg_match:
        background = bg_match.group(1).strip()
    return {
        "frames": layout,
        "background": background
    }

size = (480, 720)
size_h, size_w = size

# Color palette: consistent per object index
def get_color_by_index(index):
    rng = np.random.default_rng(index)
    return rng.random((1, 3)) * 0.6 + 0.4  # Slightly vivid colors

# Drawing utility functions
def draw_boxes(condition, frame_index=None):
    boxes, phrases = condition["boxes"], condition["phrases"]
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    colors = []
    for box_ind, (box, name) in enumerate(zip(boxes, phrases)):
        if isinstance(box, dict):
            if frame_index not in box:
                continue
        else:
            if frame_index >= len(box):
                continue
        box = box[frame_index] if frame_index is not None else box
        name = (
            name[frame_index]
            if frame_index is not None and isinstance(name, (dict, list, tuple))
            else name
        )
        c = get_color_by_index(box_ind)
        [bbox_x, bbox_y, bbox_w, bbox_h] = box
        bbox_x_max = bbox_x + bbox_w
        bbox_y_max = bbox_y + bbox_h
        if bbox_x_max <= bbox_x or bbox_y_max <= bbox_y:
            continue
        bbox_x, bbox_y, bbox_x_max, bbox_y_max = (
            bbox_x,
            bbox_y,
            bbox_x_max,
            bbox_y_max,
        )
        poly = [
            [bbox_x, bbox_y],
            [bbox_x, bbox_y_max],
            [bbox_x_max, bbox_y_max],
            [bbox_x_max, bbox_y],
        ]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        colors.append(c)
        ax.text(
            bbox_x,
            bbox_y,
            name,
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 2},
        )
    p = PatchCollection(polygons, facecolor="none", edgecolors=colors, linewidths=2)
    ax.add_collection(p)

def show_boxes(condition, frame_index=None, show=False):
    if not condition["boxes"]:
        return
    I = np.ones((size[0] + 4, size[1] + 4, 3), dtype=np.uint8) * 255
    plt.imshow(I)
    plt.axis("off")
    ax = plt.gca()
    prompt = condition.get("prompt", "")
    if prompt:
        ax.text(0, 0, prompt, style="italic", bbox={"facecolor": "white", "alpha": 0.7, "pad": 5})
    poly = [
        [0, 0],
        [0, size[0]],
        [size[1], size[0]],
        [size[1], 0],
    ]
    np_poly = np.array(poly).reshape((4, 2))
    p = PatchCollection([Polygon(np_poly)], facecolor="none", edgecolors=[(0, 0, 0)], linewidths=2)
    ax.add_collection(p)
    draw_boxes(condition, frame_index)
    if show:
        plt.show()

def generate_gif_from_layout(layout_dict, save_path="layout.gif", fps=6):
    frames_data = layout_dict["frames"]
    background = layout_dict.get("background", "")
    boxes = []
    phrases = []

    for frame_idx in sorted(frames_data.keys(), key=int):
        for obj in frames_data[frame_idx]:
            obj_id = obj["id"]
            name = obj["name"]
            box = obj["box"]
            while len(boxes) <= obj_id:
                boxes.append([])
                phrases.append([])
            boxes[obj_id].append(box)
            phrases[obj_id].append(name)

    condition = {
        "boxes": boxes,
        "phrases": phrases,
        "prompt": background
    }

    frames = []
    num_frames = len(frames_data)
    for frame_index in range(num_frames):
        fig = plt.figure(figsize=(7.2, 4.8))  # Match 720x480
        show_boxes(condition, frame_index=frame_index, show=False)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        frames.append(data)

    imageio.mimsave(save_path, frames, format="GIF", duration=1000 / fps, loop=0)
    return save_path



def convert_box(box, height, width):
    x_min, y_min = box[0] / width, box[1] / height
    w_box, h_box = box[2] / width, box[3] / height
    x_max, y_max = x_min + w_box, y_min + h_box
    return [x_min, y_min, x_max, y_max]

def interpolate_layout_boxes(layout_dict, num_interp_frames, image_size=(480, 720)):
    frames = layout_dict["frames"]
    height, width = image_size
    object_tracks = {}

    # Collect box sequences per object_id
    for frame_idx in sorted(frames.keys(), key=int):
        for obj in frames[frame_idx]:
            obj_id = obj["id"]
            box = convert_box(obj["box"], height, width)
            if obj_id not in object_tracks:
                object_tracks[obj_id] = []
            object_tracks[obj_id].append((int(frame_idx), box))

    interpolated_boxes = []

    # For each object, interpolate its boxes over desired number of frames
    for obj_id in sorted(object_tracks.keys()):
        track = object_tracks[obj_id]
        frame_indices, boxes = zip(*track)
        boxes = np.array(boxes)  # shape: (num_keyframes, 4)

        # Interpolate each coordinate separately
        interp_box = []
        for i in range(4):  # x_min, y_min, x_max, y_max
            f = interp1d(frame_indices, boxes[:, i], kind='linear')
            interp_values = f(np.linspace(min(frame_indices), max(frame_indices), num=num_interp_frames))
            interp_values = np.clip(interp_values, 0.0, 1.0)  # Clip values to [0, 1]
            interp_box.append(interp_values)

        # Transpose to shape (num_interp_frames, 4)
        interp_box = np.stack(interp_box, axis=1)
        interpolated_boxes.append(interp_box.tolist())

    return interpolated_boxes



if __name__ == "__main__":
    generate_layout_with_llm()
