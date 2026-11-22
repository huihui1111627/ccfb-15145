"""张量并行训练框架（numpy 模拟多卡 + 模拟集合通信）。"""
from .mesh import DeviceMesh
from .model import ParallelTransformer
from .params import make_configs, make_params
from .reference import RefTransformer

__all__ = [
    "DeviceMesh",
    "ParallelTransformer",
    "RefTransformer",
    "make_params",
    "make_configs",
]
