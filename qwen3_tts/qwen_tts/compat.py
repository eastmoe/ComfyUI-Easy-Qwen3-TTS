# coding=utf-8
"""Compatibility helpers for dependency APIs that move between versions."""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

_MIN_TRANSFORMERS_VERSION = Version("4.57")


def _require_transformers() -> None:
    try:
        installed = Version(version("transformers"))
    except PackageNotFoundError as exc:
        raise ImportError("Qwen-TTS requires transformers>=4.57,<6.") from exc

    if installed < _MIN_TRANSFORMERS_VERSION or installed.major >= 6:
        raise ImportError(
            "Qwen-TTS supports transformers>=4.57,<6. "
            f"Found transformers=={installed}."
        )


_require_transformers()

from transformers.cache_utils import Cache, DynamicCache  # noqa: E402
from transformers.activations import ACT2FN  # noqa: E402
from transformers.configuration_utils import PretrainedConfig  # noqa: E402
from transformers.configuration_utils import layer_type_validation  # noqa: E402
from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
from transformers.generation import GenerationMixin  # noqa: E402
from transformers.integrations import use_kernel_forward_from_hub  # noqa: E402
from transformers.masking_utils import (  # noqa: E402
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs  # noqa: E402
from transformers.modeling_layers import GradientCheckpointingLayer  # noqa: E402
from transformers.modeling_outputs import (  # noqa: E402
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import (  # noqa: E402
    ROPE_INIT_FUNCTIONS,
    dynamic_rope_update,
    rope_config_validation,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel  # noqa: E402
from transformers.models.mimi.configuration_mimi import MimiConfig  # noqa: E402
from transformers.models.mimi.modeling_mimi import MimiModel  # noqa: E402
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack  # noqa: E402
from transformers.utils import auto_docstring, can_return_tuple, logging  # noqa: E402
from transformers.utils.deprecation import deprecate_kwarg  # noqa: E402
from transformers.utils.generic import check_model_inputs  # noqa: E402
from transformers.utils.hub import cached_file  # noqa: E402

__all__ = [
    "ALL_ATTENTION_FUNCTIONS",
    "ACT2FN",
    "BaseModelOutputWithPast",
    "BatchFeature",
    "Cache",
    "CausalLMOutputWithPast",
    "DynamicCache",
    "FlashAttentionKwargs",
    "GenerationMixin",
    "GradientCheckpointingLayer",
    "MimiConfig",
    "MimiModel",
    "ModelOutput",
    "ProcessingKwargs",
    "ProcessorMixin",
    "PreTrainedModel",
    "PretrainedConfig",
    "ROPE_INIT_FUNCTIONS",
    "Unpack",
    "auto_docstring",
    "cached_file",
    "can_return_tuple",
    "check_model_inputs",
    "create_causal_mask",
    "create_sliding_window_causal_mask",
    "deprecate_kwarg",
    "dynamic_rope_update",
    "layer_type_validation",
    "logging",
    "rope_config_validation",
    "use_kernel_forward_from_hub",
]
