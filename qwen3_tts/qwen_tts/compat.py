# coding=utf-8
"""Compatibility helpers for dependency APIs that move between versions."""

from importlib.metadata import PackageNotFoundError, version
from inspect import Parameter, signature

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
try:  # noqa: E402
    from transformers.configuration_utils import ALLOWED_LAYER_TYPES  # noqa: E402
except ImportError:  # pragma: no cover - older transformers
    ALLOWED_LAYER_TYPES = None
from transformers.configuration_utils import layer_type_validation as _transformers_layer_type_validation  # noqa: E402
from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
from transformers.generation import GenerationMixin  # noqa: E402
from transformers.integrations import use_kernel_forward_from_hub  # noqa: E402
from transformers.masking_utils import (  # noqa: E402
    create_causal_mask as _transformers_create_causal_mask,
    create_sliding_window_causal_mask as _transformers_create_sliding_window_causal_mask,
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
    rope_config_validation as _transformers_rope_config_validation,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel  # noqa: E402
from transformers.models.mimi.configuration_mimi import MimiConfig  # noqa: E402
from transformers.models.mimi.modeling_mimi import MimiModel  # noqa: E402
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack  # noqa: E402
from transformers.utils import auto_docstring, can_return_tuple, logging  # noqa: E402
from transformers.utils.deprecation import deprecate_kwarg  # noqa: E402
try:  # noqa: E402
    from transformers.utils.generic import merge_with_config_defaults as _merge_with_config_defaults  # noqa: E402
except ImportError:  # pragma: no cover - older transformers
    _merge_with_config_defaults = None
from transformers.utils.generic import check_model_inputs as _transformers_check_model_inputs  # noqa: E402
from transformers.utils.hub import cached_file  # noqa: E402


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
    if hasattr(config, "standardize_rope_params"):
        config.standardize_rope_params()
    rope_parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    if layer_type is not None and isinstance(rope_parameters.get(layer_type), dict):
        rope_parameters = rope_parameters[layer_type]

    rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0)))
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    import torch

    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)


def _call_mask_function(mask_function, **kwargs):
    parameters = signature(mask_function).parameters
    accepts_kwargs = any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())
    if "inputs_embeds" in parameters and "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
        kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
    elif "input_embeds" in parameters and "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
        kwargs["input_embeds"] = kwargs.pop("inputs_embeds")

    if not accepts_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return mask_function(**kwargs)


def create_causal_mask(**kwargs):
    return _call_mask_function(_transformers_create_causal_mask, **kwargs)


def create_sliding_window_causal_mask(**kwargs):
    return _call_mask_function(_transformers_create_sliding_window_causal_mask, **kwargs)


def check_model_inputs(func=None):
    """Accept both decorator styles used across transformers releases."""
    if _merge_with_config_defaults is not None:
        return _merge_with_config_defaults if func is None else _merge_with_config_defaults(func)

    if func is not None:
        try:
            return _transformers_check_model_inputs(func)
        except TypeError:
            return _transformers_check_model_inputs()(func)

    try:
        return _transformers_check_model_inputs()
    except TypeError:
        return _transformers_check_model_inputs


def layer_type_validation(layer_types, num_hidden_layers=None, attention=True):
    """Validate layer types without triggering transformers deprecation warnings."""
    if ALLOWED_LAYER_TYPES is None:
        return _transformers_layer_type_validation(layer_types, num_hidden_layers, attention)

    if layer_types is None:
        return
    if not all(layer_type in ALLOWED_LAYER_TYPES for layer_type in layer_types):
        raise ValueError(f"The `layer_types` entries must be in {ALLOWED_LAYER_TYPES} but got {layer_types}")
    if num_hidden_layers is not None and num_hidden_layers != len(layer_types):
        raise ValueError(
            f"`num_hidden_layers` ({num_hidden_layers}) must be equal to the number of layer types "
            f"({len(layer_types)})"
        )


def rope_config_validation(config):
    """Validate RoPE parameters using the non-deprecated API when available."""
    standardize_rope_params = getattr(config, "standardize_rope_params", None)
    validate_rope = getattr(config, "validate_rope", None)
    if standardize_rope_params is not None and validate_rope is not None:
        standardize_rope_params()
        return validate_rope()
    return _transformers_rope_config_validation(config)

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
