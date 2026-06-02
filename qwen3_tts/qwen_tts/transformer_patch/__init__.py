"""Local Transformers API patches pinned to Qwen3-TTS upstream behavior."""

from .generation_4573 import Qwen3TTSGenerationMixin

__all__ = ["Qwen3TTSGenerationMixin"]
