# Qwen3-TTS Inference

This package contains the inference-only Qwen3-TTS Python implementation.
It keeps the model, tokenizer, and Python generation wrappers, and omits
training, fine-tuning, evaluation, and Web UI/GUI code.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

FlashAttention 2 is optional and can reduce GPU memory usage when your hardware
and PyTorch environment support it.

```bash
pip install -U flash-attn --no-build-isolation
```

## Custom Voice

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese",
    speaker="Vivian",
    instruct="用特别愤怒的语气说",
)
sf.write("output_custom_voice.wav", wavs[0], sr)
```

## Voice Design

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = model.generate_voice_design(
    text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。",
)
sf.write("output_voice_design.wav", wavs[0], sr)
```

## Voice Clone

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = model.generate_voice_clone(
    text="She said she would be here by noon.",
    language="English",
    ref_audio="reference.wav",
    ref_text="Reference transcript for the uploaded audio.",
)
sf.write("output_voice_clone.wav", wavs[0], sr)
```

## Tokenizer

```python
import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
)

encoded = tokenizer.encode("input.wav")
wavs, sr = tokenizer.decode(encoded)
sf.write("decoded.wav", wavs[0], sr)
```
