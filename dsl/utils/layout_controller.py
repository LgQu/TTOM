# layout_controller.py

import os, json
from typing import Dict, List, Tuple
from motion import MOTION_REGISTRY
from layout_handler import generate_gif_from_layout

class Motion:
    def __init__(self, start_frame: int, end_frame: int,
                 start_pos: Tuple[int, int], end_pos: Tuple[int, int],
                 motion_type: str = "linear", params: dict = {}):
        self.start = start_frame
        self.end = end_frame
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.motion_type = motion_type
        self.params = params

    def apply_to_box(self, prev_box: Tuple[int, int, int, int], frame: int) -> Tuple[int, int, int, int]:
        if not (self.start <= frame <= self.end):
            return prev_box

        cx, cy, w, h = prev_box
        progress = (frame - self.start) / max(1, self.end - self.start)
        fn = MOTION_REGISTRY.get(self.motion_type, lambda p, _: p)
        alpha = fn(progress, self.params)

        if self.motion_type != "scale":
            cx = int(self.start_pos[0] + alpha * (self.end_pos[0] - self.start_pos[0]))
            cy = int(self.start_pos[1] + alpha * (self.end_pos[1] - self.start_pos[1]))

        if self.motion_type == "scale":
            sw, sh = self.params["start_size"]
            ew, eh = self.params["end_size"]
            w = int(sw + alpha * (ew - sw))
            h = int(sh + alpha * (eh - sh))

        return cx, cy, w, h



class FrameBox:
    def __init__(self, cx: int, cy: int, w: int, h: int):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.motions: List[Motion] = []


class Box:
    def __init__(self, box_id: int, name: str, cx: int, cy: int, w: int, h: int, frames_duration: List[int]):
        self.id = box_id
        self.name = name
        self.init_cx = cx
        self.init_cy = cy
        self.init_w = w
        self.init_h = h
        self.frames: Dict[int, FrameBox] = {
            f: FrameBox(cx, cy, w, h) for f in frames_duration
        }
        self.frames_duration = frames_duration

    def bind_motion(self, motion: Motion):
        for f in range(motion.start, motion.end + 1):
            self.frames[f].motions.append(motion)
        self.update_all_frames()

    def update_all_frames(self):
        last_cx, last_cy, last_w, last_h = self.init_cx, self.init_cy, self.init_w, self.init_h
        for f in sorted(self.frames.keys()):
            frame_box = self.frames[f]
            cx, cy, w, h = last_cx, last_cy, last_w, last_h

            for motion in frame_box.motions:
                cx, cy, w, h = motion.apply_to_box((cx, cy, w, h), f)

            frame_box.cx, frame_box.cy = cx, cy
            frame_box.w, frame_box.h = w, h

            last_cx, last_cy, last_w, last_h = cx, cy, w, h
    
    def get_box(self, frame: int):
        return self.frames[frame].cx, self.frames[frame].cy, self.frames[frame].w, self.frames[frame].h

class LayoutController:
    def __init__(self, width=720, height=480, total_frames=13):
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.objects: Dict[str, Box] = {}
        self.layout_data = {}

    def register_object(self, box_id: int, name: str, cx: int, cy: int, w: int, h: int, frames_duration=None):
        if frames_duration is None:
            frames_duration = self.get_frames_duration()
        self.objects[box_id] = Box(box_id, name, cx, cy, w, h, frames_duration)

    def bind_motion(self, box_id: int, start_frame: int, end_frame: int,
                    start_pos: Tuple[int, int], end_pos: Tuple[int, int],
                    motion_type: str = "linear", params: dict = {}):
        motion = Motion(start_frame, end_frame, start_pos, end_pos, motion_type, params)
        self.objects[box_id].bind_motion(motion)
    
    def get_frames_duration(self):
        frames_duration = [i for i in range(1, self.total_frames + 1)]
        return frames_duration

    def update_all(self):
        for box in self.objects.values():
            box.update_all_frames()

    def render_to_json(self, json_path: str):
        layout_data = {}

        for frame in range(1, self.total_frames + 1):
            frame_data = []
            for box in self.objects.values():
                if frame in box.frames:
                    cx, cy, w, h = box.get_box(frame)
                    x = cx - w // 2
                    y = cy - h // 2
                    frame_data.append({
                        "id": box.id,
                        "name": box.name,
                        "box": [x, y, w, h]
                    })
            layout_data[str(frame)] = frame_data

        with open(json_path, "w") as f:
            json.dump({"layout": {"frames": layout_data}}, f, indent=2)

        self.layout_data = {"frames": layout_data}
        return self.layout_data


    def render_to_video(self, video_path: str):
        generate_gif_from_layout(self.layout_data, save_path=video_path)

if __name__ == "__main__":
    ctrl = LayoutController()
    ctrl.register_object(0, "box1", cx=160, cy=240, w=120, h=80)

    ctrl.bind_motion(0, start_frame=1, end_frame=10,
                    start_pos=(100, 200), end_pos=(400, 100),
                    motion_type="ease_out_quad")

    out_dir = "/hpctmp/e1351271/lvd/LVD_extention/data/layout"
    ctrl.render_to_json(json_path=f"{out_dir}/layout.json")
    ctrl.render_to_video(video_path=f"{out_dir}/layout.gif")