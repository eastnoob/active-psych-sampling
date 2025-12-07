"""
Adapters for Subject Simulator V2

提供与其他系统兼容的适配器接口
"""

from .warmup_adapter import run, simulate_responses

__all__ = ["run", "simulate_responses"]
