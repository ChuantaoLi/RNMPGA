"""
模型注册表
"""

from .rnmpga import jfs as rnmpga_jfs

MODELS = {
    "rnmpga": rnmpga_jfs,
}


def get_model(name: str):
    if name not in MODELS:
        raise KeyError(f"未知模型 '{name}', 已注册模型: {list(MODELS)}")
    return MODELS[name]
