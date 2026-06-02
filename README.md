# ComfyUI-Easy-Qwen3-TTS
Qwen3-TTS在comfyui里的工程实现

## 节点

右键菜单位置：

```text
eastmoe -> Comfy-Easy-Qwen3-TTS
```

插件提供 1 个加载节点和 3 个独立推理节点：

- `Qwen3-TTS 加载模型`：加载 Custom Voice、Voice Design 或 Voice Clone/Base 模型，输出 `QWEN3_TTS_MODEL`。
- `Qwen3-TTS 预置音色合成`：对应 `generate_custom_voice`，需要 Custom Voice 模型。
- `Qwen3-TTS 音色设计合成`：对应 `generate_voice_design`，需要 Voice Design 模型。
- `Qwen3-TTS 音色克隆合成`：对应 `generate_voice_clone`，需要 Base 模型和 ComfyUI `AUDIO` 参考音频。

## 模型目录

默认模型根目录为：

```text
ComfyUI/models/qwen3-tts
```

建议按主模型和依赖模型分别放置子目录：

```text
ComfyUI/models/qwen3-tts/
  custom-voice/
  voice-design/
  base/
  speech-tokenizer/
```

加载节点的 `模型路径` 填 `auto` 时会按 `模型模式` 自动选择：

- `custom-voice` -> `ComfyUI/models/qwen3-tts/custom-voice`
- `voice-design` -> `ComfyUI/models/qwen3-tts/voice-design`
- `voice-clone` -> `ComfyUI/models/qwen3-tts/base`

Qwen3-TTS 当前推理代码会在主模型目录下查找 `speech_tokenizer/config.json`。为了同时满足依赖模型独立子目录的放置方式，加载节点默认会在主模型缺少 `speech_tokenizer` 时，将 `ComfyUI/models/qwen3-tts/speech-tokenizer` 链接到主模型目录下；如果系统不允许创建链接，会退化为复制目录。

## 加载设置

加载节点支持设置：

- `计算精度`：`auto`、`float16`、`bfloat16`、`float32`
- `计算设备`：`cuda:0`、`auto`、`cpu`
- `注意力实现`：`auto`、`flash_attention_2`、`sdpa`、`eager`
- `仅使用本地文件`：默认开启，适合离线模型目录

如果希望 `auto` 目录不存在时临时走 Hugging Face 仓库，可关闭 `仅使用本地文件` 并开启 `允许仓库回退`。

## 中文界面

中文翻译位于：

```text
local/zh-cn/nodes.json
```

插件同时注册前端扩展，会在节点加载和创建时读取该文件并覆盖节点名称、参数标签和输出接口显示。
