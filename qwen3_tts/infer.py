# coding=utf-8
"""Command line inference utility for Qwen3-TTS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_MODEL_BY_MODE = {
    "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice-clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


def _parse_jsonish(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered == "none":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _parse_key_value(items: Optional[Iterable[str]], option_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"{option_name} expects KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"{option_name} got an empty key in: {item!r}")
        out[key] = _parse_jsonish(value)
    return out


def _parse_dtype(value: Optional[str]):
    if value is None or value == "auto":
        return None
    import torch

    dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    try:
        return dtypes[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {value}") from exc


def _read_text_file(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def _as_list(values: Optional[List[str]], file_value: Optional[str] = None) -> Optional[List[str]]:
    out: List[str] = []
    if values:
        out.extend(values)
    if file_value is not None:
        out.append(file_value)
    return out or None


def _coerce_scalar_or_list(values: Optional[List[Any]]) -> Any:
    if values is None:
        return None
    return values[0] if len(values) == 1 else values


def _parse_bool_list(values: Optional[List[str]]) -> Optional[List[bool]]:
    if not values:
        return None
    out = []
    for value in values:
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            out.append(True)
        elif lowered in {"0", "false", "f", "no", "n", "off"}:
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value!r}")
    return out


def _model_load_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs = _parse_key_value(args.model_kwarg, "--model-kwarg")
    dtype = _parse_dtype(args.dtype)
    if dtype is not None:
        kwargs["dtype"] = dtype
    for arg_name, kwarg_name in (
        ("device_map", "device_map"),
        ("attn_implementation", "attn_implementation"),
        ("cache_dir", "cache_dir"),
        ("revision", "revision"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            kwargs[kwarg_name] = value
    for arg_name, kwarg_name in (
        ("local_files_only", "local_files_only"),
        ("low_cpu_mem_usage", "low_cpu_mem_usage"),
        ("use_safetensors", "use_safetensors"),
        ("trust_remote_code", "trust_remote_code"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            kwargs[kwarg_name] = value
    return kwargs


def _generate_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs = _parse_key_value(args.generate_kwarg, "--generate-kwarg")
    for name in (
        "do_sample",
        "top_k",
        "top_p",
        "temperature",
        "repetition_penalty",
        "subtalker_dosample",
        "subtalker_top_k",
        "subtalker_top_p",
        "subtalker_temperature",
        "max_new_tokens",
    ):
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value
    return kwargs


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_output_paths(output: str, count: int, overwrite: bool) -> List[Path]:
    output_path = Path(output)
    paths: List[Path] = []

    if count == 1 and output_path.suffix:
        paths = [output_path]
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        paths = [output_path / f"output_{i:03d}.wav" for i in range(count)]

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite to replace it: {path}")
    return paths


def _save_wavs(wavs: List[np.ndarray], sample_rate: int, output: str, overwrite: bool) -> List[Path]:
    import numpy as np
    import torch
    import torchaudio

    paths = _safe_output_paths(output, len(wavs), overwrite=overwrite)
    for wav, path in zip(wavs, paths):
        waveform = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
        torchaudio.save(str(path), waveform, int(sample_rate))
    return paths


def _print_saved(paths: List[Path], sample_rate: int) -> None:
    print(f"Saved {len(paths)} audio file(s) at {sample_rate} Hz:")
    for path in paths:
        print(f"  {path}")


def _load_tts_model(args: argparse.Namespace, default_model: str) -> Qwen3TTSModel:
    from qwen_tts import Qwen3TTSModel

    model_path = args.model or default_model
    print(f"Loading model: {model_path}")
    return Qwen3TTSModel.from_pretrained(model_path, **_model_load_kwargs(args))


def _run_custom_voice(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    model = _load_tts_model(args, DEFAULT_MODEL_BY_MODE["custom-voice"])
    texts = _as_list(args.text, _read_text_file(args.text_file))
    if not texts:
        raise ValueError("custom-voice requires --text or --text-file.")
    wavs, sample_rate = model.generate_custom_voice(
        text=_coerce_scalar_or_list(texts),
        language=_coerce_scalar_or_list(args.language),
        speaker=_coerce_scalar_or_list(args.speaker),
        instruct=_coerce_scalar_or_list(args.instruct),
        **({"non_streaming_mode": args.non_streaming_mode} if args.non_streaming_mode is not None else {}),
        **_generate_kwargs(args),
    )
    _print_saved(_save_wavs(wavs, sample_rate, args.output, args.overwrite), sample_rate)


def _run_voice_design(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    model = _load_tts_model(args, DEFAULT_MODEL_BY_MODE["voice-design"])
    texts = _as_list(args.text, _read_text_file(args.text_file))
    instructs = _as_list(args.instruct, _read_text_file(args.instruct_file))
    if not texts:
        raise ValueError("voice-design requires --text or --text-file.")
    if not instructs:
        raise ValueError("voice-design requires --instruct or --instruct-file.")
    wavs, sample_rate = model.generate_voice_design(
        text=_coerce_scalar_or_list(texts),
        language=_coerce_scalar_or_list(args.language),
        instruct=_coerce_scalar_or_list(instructs),
        **({"non_streaming_mode": args.non_streaming_mode} if args.non_streaming_mode is not None else {}),
        **_generate_kwargs(args),
    )
    _print_saved(_save_wavs(wavs, sample_rate, args.output, args.overwrite), sample_rate)


def _run_voice_clone(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    model = _load_tts_model(args, DEFAULT_MODEL_BY_MODE["voice-clone"])
    texts = _as_list(args.text, _read_text_file(args.text_file))
    ref_texts = _as_list(args.ref_text, _read_text_file(args.ref_text_file))
    if not texts:
        raise ValueError("voice-clone requires --text or --text-file.")
    if not args.ref_audio:
        raise ValueError("voice-clone requires --ref-audio.")

    x_vector_modes = _parse_bool_list(args.x_vector_only_modes)
    x_vector_only_mode: Any
    if x_vector_modes is not None:
        x_vector_only_mode = _coerce_scalar_or_list(x_vector_modes)
    else:
        x_vector_only_mode = bool(args.x_vector_only_mode)

    wavs, sample_rate = model.generate_voice_clone(
        text=_coerce_scalar_or_list(texts),
        language=_coerce_scalar_or_list(args.language),
        ref_audio=_coerce_scalar_or_list(args.ref_audio),
        ref_text=_coerce_scalar_or_list(ref_texts),
        x_vector_only_mode=x_vector_only_mode,
        **({"non_streaming_mode": args.non_streaming_mode} if args.non_streaming_mode is not None else {}),
        **_generate_kwargs(args),
    )
    _print_saved(_save_wavs(wavs, sample_rate, args.output, args.overwrite), sample_rate)


def _run_inspect(args: argparse.Namespace) -> None:
    model = _load_tts_model(args, args.model or DEFAULT_MODEL_BY_MODE["custom-voice"])
    attrs = {
        "tokenizer_type": getattr(model.model, "tokenizer_type", None),
        "tts_model_size": getattr(model.model, "tts_model_size", None),
        "tts_model_type": getattr(model.model, "tts_model_type", None),
        "supported_languages": model.get_supported_languages(),
        "supported_speakers": model.get_supported_speakers(),
    }
    print(json.dumps(attrs, ensure_ascii=False, indent=2))


def _load_tokenizer(args: argparse.Namespace) -> Qwen3TTSTokenizer:
    from qwen_tts import Qwen3TTSTokenizer

    model_path = args.tokenizer_model or DEFAULT_TOKENIZER_MODEL
    print(f"Loading tokenizer: {model_path}")
    return Qwen3TTSTokenizer.from_pretrained(model_path, **_model_load_kwargs(args))


def _save_encoded_npz(encoded: Any, output: str, overwrite: bool) -> Path:
    import numpy as np

    path = Path(output)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    audio_codes = encoded.audio_codes
    xvectors = getattr(encoded, "xvectors", None)
    ref_mels = getattr(encoded, "ref_mels", None)
    count = len(audio_codes)

    arrays: Dict[str, Any] = {
        "manifest": np.array(json.dumps({"version": 1, "count": count}), dtype=np.str_),
    }
    for i in range(count):
        arrays[f"sample_{i}_audio_codes"] = audio_codes[i].detach().cpu().numpy()
        if xvectors is not None:
            arrays[f"sample_{i}_xvectors"] = xvectors[i].detach().cpu().numpy()
        if ref_mels is not None:
            arrays[f"sample_{i}_ref_mels"] = ref_mels[i].detach().cpu().numpy()
    np.savez_compressed(path, **arrays)
    return path


def _load_encoded_npz(path: str) -> List[Dict[str, np.ndarray]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        count = int(manifest["count"])
        encoded = []
        for i in range(count):
            item: Dict[str, np.ndarray] = {
                "audio_codes": data[f"sample_{i}_audio_codes"],
            }
            x_key = f"sample_{i}_xvectors"
            mel_key = f"sample_{i}_ref_mels"
            if x_key in data:
                item["xvectors"] = data[x_key]
            if mel_key in data:
                item["ref_mels"] = data[mel_key]
            encoded.append(item)
    return encoded


def _run_tokenizer_encode(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer(args)
    if not args.audio:
        raise ValueError("tokenizer-encode requires --audio.")
    encoded = tokenizer.encode(_coerce_scalar_or_list(args.audio), sr=args.sr, return_dict=True)
    path = _save_encoded_npz(encoded, args.output, args.overwrite)
    print(f"Saved encoded tokenizer output: {path}")


def _run_tokenizer_decode(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer(args)
    encoded = _load_encoded_npz(args.input)
    wavs, sample_rate = tokenizer.decode(encoded)
    _print_saved(_save_wavs(wavs, sample_rate, args.output, args.overwrite), sample_rate)


def _add_model_args(parser: argparse.ArgumentParser, *, tokenizer: bool = False) -> None:
    if tokenizer:
        parser.add_argument("--tokenizer-model", help="Tokenizer repo id or local path.")
    else:
        parser.add_argument("--model", help="Model repo id or local path.")
    parser.add_argument("--device-map", help='Device map passed to from_pretrained, e.g. "cuda:0", "auto", "cpu".')
    parser.add_argument("--dtype", default="auto", help="auto, float16/fp16, bfloat16/bf16, float32/fp32, float64/fp64.")
    parser.add_argument("--attn-implementation", help='Attention implementation, e.g. "flash_attention_2", "sdpa", "eager".')
    parser.add_argument("--cache-dir", help="HuggingFace cache directory.")
    parser.add_argument("--revision", help="Model revision.")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-safetensors", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--model-kwarg",
        action="append",
        help="Extra from_pretrained kwarg as KEY=VALUE. VALUE is parsed as JSON when possible.",
    )


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--subtalker-dosample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--subtalker-top-k", type=int)
    parser.add_argument("--subtalker-top-p", type=float)
    parser.add_argument("--subtalker-temperature", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--non-streaming-mode", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--generate-kwarg",
        action="append",
        help="Extra model.generate kwarg as KEY=VALUE. VALUE is parsed as JSON when possible.",
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", required=True, help="Output wav file for one sample, or directory for a batch.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")


def _add_text_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text", action="append", help="Text to synthesize. Repeat for batch inference.")
    parser.add_argument("--text-file", help="UTF-8 text file to synthesize.")
    parser.add_argument("--language", action="append", help="Language name. Repeat to match --text, or pass once for all.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS CLI inference utility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    custom = subparsers.add_parser("custom-voice", help="Generate with a CustomVoice model.")
    _add_model_args(custom)
    _add_generation_args(custom)
    _add_output_args(custom)
    _add_text_args(custom)
    custom.add_argument("--speaker", action="append", required=True, help="Speaker name. Repeat for batch, or pass once for all.")
    custom.add_argument("--instruct", action="append", help="Optional instruction. Repeat for batch.")
    custom.set_defaults(func=_run_custom_voice)

    design = subparsers.add_parser("voice-design", help="Generate with a VoiceDesign model.")
    _add_model_args(design)
    _add_generation_args(design)
    _add_output_args(design)
    _add_text_args(design)
    design.add_argument("--instruct", action="append", help="Voice/style instruction. Repeat for batch.")
    design.add_argument("--instruct-file", help="UTF-8 instruction file.")
    design.set_defaults(func=_run_voice_design)

    clone = subparsers.add_parser("voice-clone", help="Generate with a Base voice clone model.")
    _add_model_args(clone)
    _add_generation_args(clone)
    _add_output_args(clone)
    _add_text_args(clone)
    clone.add_argument("--ref-audio", action="append", required=True, help="Reference audio path, URL, or base64. Repeat for batch.")
    clone.add_argument("--ref-text", action="append", help="Reference transcript for ICL mode. Repeat for batch.")
    clone.add_argument("--ref-text-file", help="UTF-8 reference transcript file.")
    clone.add_argument("--x-vector-only-mode", action="store_true", help="Use speaker embedding only for all samples.")
    clone.add_argument(
        "--x-vector-only-modes",
        action="append",
        help="Per-reference x-vector-only boolean. Repeat using true/false values.",
    )
    clone.set_defaults(func=_run_voice_clone)

    inspect = subparsers.add_parser("inspect", help="Print model type and supported languages/speakers.")
    _add_model_args(inspect)
    inspect.set_defaults(func=_run_inspect)

    encode = subparsers.add_parser("tokenizer-encode", help="Encode audio to a portable .npz tokenizer payload.")
    _add_model_args(encode, tokenizer=True)
    encode.add_argument("--audio", action="append", required=True, help="Audio path, URL, or base64. Repeat for batch.")
    encode.add_argument("--sr", type=int, help="Original sample rate for raw numpy inputs; unused for paths/URLs/base64.")
    encode.add_argument("-o", "--output", required=True, help="Output .npz path.")
    encode.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    encode.set_defaults(func=_run_tokenizer_encode)

    decode = subparsers.add_parser("tokenizer-decode", help="Decode a .npz tokenizer payload to waveform.")
    _add_model_args(decode, tokenizer=True)
    decode.add_argument("-i", "--input", required=True, help="Input .npz path from tokenizer-encode.")
    _add_output_args(decode)
    decode.set_defaults(func=_run_tokenizer_decode)

    return parser


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
