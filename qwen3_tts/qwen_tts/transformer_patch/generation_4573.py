# coding=utf-8
"""Generation helpers localized from transformers==4.57.3 for Qwen3-TTS."""

from __future__ import annotations

from .transformers_4573.generation.configuration_utils import GenerationConfig
from .transformers_4573.generation.utils import GenerationMixin


class Qwen3TTSGenerationMixin(GenerationMixin):
    """Transformers 4.57.3 generation path used by Qwen3-TTS talker modules."""

    def generate(self, *args, generation_config=None, **kwargs):
        if generation_config is None:
            base_config = getattr(self, "generation_config", None)
            if base_config is not None and hasattr(base_config, "to_dict"):
                generation_config = GenerationConfig.from_dict(
                    {key: value for key, value in base_config.to_dict().items() if value is not None}
                )
            else:
                generation_config = GenerationConfig()
        elif not isinstance(generation_config, GenerationConfig):
            generation_config = GenerationConfig.from_dict(
                {key: value for key, value in generation_config.to_dict().items() if value is not None}
            )
        kwargs.setdefault("use_model_defaults", False)
        return super().generate(*args, generation_config=generation_config, **kwargs)
