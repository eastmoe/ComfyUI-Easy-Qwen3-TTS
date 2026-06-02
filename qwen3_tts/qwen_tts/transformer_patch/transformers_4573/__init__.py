# coding=utf-8
"""Vendored Transformers 4.57.3 subset for Qwen3-TTS generation.

This package is intentionally imported under qwen_tts.transformer_patch, not as
top-level `transformers`, so it does not replace ComfyUI's global Transformers
installation.  The top-level initializer is kept minimal to avoid running the
vendored dependency version checks against the host environment.
"""

__version__ = "4.57.3"

from .tokenization_utils_base import AddedToken

__all__ = ["AddedToken", "__version__"]
