# coding=utf-8
"""No-op dependency checks for the vendored Transformers 4.57.3 patch.

The vendored package is used only for Qwen3-TTS generation internals and must
not validate or reject the host ComfyUI Python environment.
"""


def dep_version_check(pkg, hint=None):
    return None
