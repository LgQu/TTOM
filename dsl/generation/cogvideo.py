"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""

import logging
import argparse
from typing import Literal, Optional
from diffusers.utils import export_to_video, load_image, load_video
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
# from cogvideo.controllable_pipeline_cogvideox import (
#     CogVideoXPipeline,
#     CogVideoXDDIMScheduler,
# )
from dsl.utils import cache_old, parse
from dsl.utils.parse import show_video_boxes
from dsl.utils.llm import get_parsed_layout
from dsl.utils.prompt import get_prompts
from dsl.utils.prompt_convert import convert_prompt

from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)

logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (1360, 768),
    "cogvideox1.5-5b": (1360, 768),

    # cogvideox-*
    "cogvideox-5b-i2v": (720, 480),
    "cogvideox-5b": (720, 480),
    "cogvideox-2b": (720, 480),
}

## Visualize
def visualize_layout(parsed_layout):
    H, W = parse.size
    condition = parse.parsed_layout_to_condition(
        parsed_layout, tokenizer=None, height=H, width=W, verbose=True
    )

    show_video_boxes(condition, ind=0, save=True)

    print(f"Visualize masks at {parse.img_dir}")

def generate_video(
    prompt_type: str,
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    # condition
    parsed_layout=None,
    # backward guidance
    loss_scale=2.5,
    loss_threshold=200.0,
    max_iter=1,
    max_index_step=10,
    fg_top_p=0.25,
    bg_top_p=0.25,
    fg_weight=1.0,
    bg_weight=2.0,
    attn_sync_weight=0.0,
    boxdiff_loss_scale=0.0,
    boxdiff_normed=True,
    com_loss_scale=0.0,
    use_ratio_based_loss=False,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP[model_name]
    if width is None or height is None:
        width, height = desired_resolution
        logging.info(f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m")
    elif (width, height) != desired_resolution:
        if generate_type == "i2v":
            # For i2v models, use user-defined width and height
            logging.warning(f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m")
        else:
            # Otherwise, use the recommended width and height
            logging.warning(f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m")
            width, height = desired_resolution

    # if generate_type == "i2v":
    #     pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    #     image = load_image(image=image_or_video_path)
    # elif generate_type == "t2v":
    #     pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    # else:
    #     pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    #     video = load_video(image_or_video_path)

    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # If you're using with lora, add this code
    # if lora_path:
    #     pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
    #     pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    # pipe.to("cuda")
    # TODO
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # if generate_type == "i2v":
    #     video_generate = pipe(
    #         height=height,
    #         width=width,
    #         prompt=prompt,
    #         image=image,
    #         # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
    #         num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
    #         num_inference_steps=num_inference_steps,  # Number of inference steps
    #         num_frames=num_frames,  # Number of frames to generate
    #         use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
    #         guidance_scale=guidance_scale,
    #         generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
    #     ).frames[0]
    # elif generate_type == "t2v":
    #     video_generate = pipe(
    #         height=height,
    #         width=width,
    #         prompt=prompt,
    #         num_videos_per_prompt=num_videos_per_prompt,
    #         num_inference_steps=num_inference_steps,
    #         num_frames=num_frames,
    #         use_dynamic_cfg=True,
    #         guidance_scale=guidance_scale,
    #         generator=torch.Generator().manual_seed(seed),
    #     ).frames[0]
    # else:
    #     video_generate = pipe(
    #         height=height,
    #         width=width,
    #         prompt=prompt,
    #         video=video,  # The path of the video to be used as the background of the video
    #         num_videos_per_prompt=num_videos_per_prompt,
    #         num_inference_steps=num_inference_steps,
    #         num_frames=num_frames,
    #         use_dynamic_cfg=True,
    #         guidance_scale=guidance_scale,
    #         generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
    #     ).frames[0]

    # * Prepare backward_guidance_kwargs

    box_H, box_W = parse.size

    # Load the parsed layout from the cache
    cache_old.cache_format = "json"
    cache_old.cache_path = f'cache/cache_{args.prompt_type.replace("lmd_", "")}_v0.1_gpt-4-1106-preview.json'
    print(f"\nLoading LLM responses from cache {cache_old.cache_path}")
    cache_old.init_cache(allow_nonexist=False)

    prompt = get_prompts(prompt_type)[0]
    resp = cache_old.get_cache(prompt)

    print(f"\nprompt: {prompt}, resp: {resp}")
    # Directly use the response from the cache
    parsed_layout, _ = get_parsed_layout(
        prompt,
        max_partial_response_retries=1,
        override_response=resp,
        json_template=False,
    )
    print("\nparsed_layout: ", parsed_layout)

    visualize_layout(parsed_layout)

    condition = parse.parsed_layout_to_condition(
        parsed_layout,
        tokenizer=pipe.tokenizer,
        height=box_H,
        width=box_W,
        num_condition_frames=num_frames,
        verbose=True,
    )
    print('\n\nCondition: ', condition)

    prompt, bboxes, phrases, object_positions, token_map = (
        condition.prompt,
        condition.boxes,
        condition.phrases,
        condition.object_positions,
        condition.token_map,
    )

    backward_guidance_kwargs = dict(
        bboxes=bboxes,
        object_positions=object_positions,
        loss_scale=loss_scale,
        loss_threshold=loss_threshold,
        max_iter=max_iter,
        max_index_step=max_index_step,
        fg_top_p=fg_top_p,
        bg_top_p=bg_top_p,
        fg_weight=fg_weight,
        bg_weight=bg_weight,
        use_ratio_based_loss=use_ratio_based_loss,
        # guidance_attn_keys=overall_guidance_attn_keys,
        exclude_bg_heads=False,
        # upsample_scale=upsample_scale,
        # upsample_mode=upsample_mode,
        # base_attn_dim=base_attn_dim,
        attn_sync_weight=attn_sync_weight,
        boxdiff_loss_scale=boxdiff_loss_scale,
        boxdiff_normed=boxdiff_normed,
        com_loss_scale=com_loss_scale,
        verbose=True,
    )

    return_guidance_saved_attn = False

    prompt = convert_prompt(prompt)

    print(f"\n\nConverted prompt: {prompt}")

    video_generate = pipe(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        
        # guidance_callback=None,
        # backward_guidance_kwargs=backward_guidance_kwargs,
        # backward_guidance_kwargs=None,
        # return_guidance_saved_attn=return_guidance_saved_attn,
        # guidance_type="main",
    ).frames[0]

    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt-type", type=str, default="demo")
    parser.add_argument("--prompt", type=str, default=None, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-2b", help="Path of the pre-trained model use"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument("--height", type=int, default=None, help="The height of the generated video")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt_type = args.prompt_type,
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
    )
