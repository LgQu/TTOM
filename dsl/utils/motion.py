# motion.py

from typing import Dict, Callable
import math

# 注册所有运动类型：名称 -> 插值函数
MOTION_REGISTRY: Dict[str, Callable[[float, Dict], float]] = {}

def register_motion(name: str):
    def wrapper(func):
        MOTION_REGISTRY[name] = func
        return func
    return wrapper

# 线性插值
@register_motion("linear")
def linear(progress: float, params: Dict) -> float:
    return progress

# 二次 ease-in（慢进快出）
@register_motion("ease_in_quad") 
def ease_in_quad(progress: float, params: Dict) -> float:
    return progress ** 2

# 二次 ease-out（快进慢出）
@register_motion("ease_out_quad")
def ease_out_quad(progress: float, params: Dict) -> float:
    return 1 - (1 - progress) ** 2
