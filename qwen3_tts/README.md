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
import torchaudio
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
torchaudio.save("output_custom_voice.wav", torch.from_numpy(wavs[0]).unsqueeze(0), sr)
```

## Voice Design

```python
import torch
import torchaudio
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
torchaudio.save("output_voice_design.wav", torch.from_numpy(wavs[0]).unsqueeze(0), sr)
```

## Voice Clone

```python
import torch
import torchaudio
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
torchaudio.save("output_voice_clone.wav", torch.from_numpy(wavs[0]).unsqueeze(0), sr)
```

## Tokenizer

```python
import torch
import torchaudio
from qwen_tts import Qwen3TTSTokenizer

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
)

encoded = tokenizer.encode("input.wav")
wavs, sr = tokenizer.decode(encoded)
torchaudio.save("decoded.wav", torch.from_numpy(wavs[0]).unsqueeze(0), sr)
```

## CLI Inference

`infer.py` exposes the same inference modes from the Python wrappers. Common
model loading arguments include `--model`, `--device-map`, `--dtype`,
`--attn-implementation`, and repeatable `--model-kwarg KEY=VALUE`. Generation
arguments include `--top-k`, `--top-p`, `--temperature`,
`--repetition-penalty`, `--max-new-tokens`, subtalker sampling arguments, and
repeatable `--generate-kwarg KEY=VALUE`.

```bash
python infer.py custom-voice \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device-map cuda:0 --dtype bfloat16 --attn-implementation flash_attention_2 \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --language Chinese \
  --speaker Vivian \
  --instruct "用特别愤怒的语气说" \
  -o output_custom_voice.wav --overwrite
```

```bash
python infer.py voice-design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --device-map cuda:0 --dtype bfloat16 \
  --text "哥哥，你回来啦，人家等了你好久好久了，要抱抱！" \
  --language Chinese \
  --instruct "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。" \
  -o output_voice_design.wav --overwrite
```

```bash
python infer.py voice-clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --device-map cuda:0 --dtype bfloat16 \
  --text "She said she would be here by noon." \
  --language English \
  --ref-audio reference.wav \
  --ref-text "Reference transcript for the uploaded audio." \
  -o output_voice_clone.wav --overwrite
```

Use speaker-embedding-only clone mode by adding `--x-vector-only-mode` and
omitting `--ref-text`.

```bash
python infer.py voice-clone \
  --text "She said she would be here by noon." \
  --language English \
  --ref-audio reference.wav \
  --x-vector-only-mode \
  -o output_voice_clone_xvec.wav --overwrite
```

Tokenizer encode/decode can be run through a portable `.npz` payload.

```bash
python infer.py tokenizer-encode \
  --tokenizer-model Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --device-map cuda:0 \
  --audio input.wav \
  -o encoded.npz --overwrite

python infer.py tokenizer-decode \
  --tokenizer-model Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --device-map cuda:0 \
  -i encoded.npz \
  -o decoded.wav --overwrite
```
