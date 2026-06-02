from __future__ import annotations

import gc
import inspect
import json
import os
import random
import shutil
import socket
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch

try:
    import comfy.model_management as comfy_model_management
    import comfy.utils as comfy_utils
except ImportError:
    comfy_model_management = None
    comfy_utils = None

try:
    import folder_paths
except ImportError:
    folder_paths = None


PLUGIN_DIR = Path(__file__).resolve().parents[2]
QWEN3_TTS_SRC_DIR = PLUGIN_DIR / "qwen3_tts"
LOCAL_DIR = PLUGIN_DIR / "local"

CATEGORY = "eastmoe/Comfy-Easy-Qwen3-TTS"
MODEL_FOLDER_KEY = "qwen3-tts"
MODEL_FOLDER_NAME = "qwen3-tts"
MODEL_TYPE = "QWEN3_TTS_MODEL"

MODEL_MODE_OPTIONS = ["custom-voice", "voice-design", "voice-clone"]
DOWNLOAD_SOURCE_OPTIONS = ["huggingface", "hf-mirror", "custom"]
PRECISION_OPTIONS = ["auto", "float16", "bfloat16", "float32"]
DEVICE_MAP_OPTIONS = ["cuda:0", "auto", "cpu"]
ATTENTION_OPTIONS = ["auto", "flash_attention_2", "sdpa", "eager"]

DEFAULT_MODEL_SUBDIRS = {
    "custom-voice": "custom-voice",
    "voice-design": "voice-design",
    "voice-clone": "base",
}
DEFAULT_REPO_IDS = {
    "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice-clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
DEFAULT_TOKENIZER_SUBDIR = "speech-tokenizer"
DEFAULT_TOKENIZER_REPO_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

_RUNTIME_LOCK = threading.RLock()
_MODEL_CACHE: dict[tuple[Any, ...], "Qwen3TTSHandle"] = {}


def _ensure_qwen_tts_on_path() -> None:
    src = str(QWEN3_TTS_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def _comfy_root() -> Path:
    return PLUGIN_DIR.parent.parent


def comfy_models_dir() -> Path:
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        return Path(folder_paths.models_dir)
    return _comfy_root() / "models"


def default_model_root() -> Path:
    return comfy_models_dir() / MODEL_FOLDER_NAME


def default_model_dir(model_mode: str) -> Path:
    return default_model_root() / DEFAULT_MODEL_SUBDIRS.get(model_mode, "custom-voice")


def default_tokenizer_dir() -> Path:
    return default_model_root() / DEFAULT_TOKENIZER_SUBDIR


def register_model_folder() -> None:
    root = default_model_root()
    root.mkdir(parents=True, exist_ok=True)
    if folder_paths is None:
        return
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path(MODEL_FOLDER_KEY, str(root), is_default=True)
    elif hasattr(folder_paths, "folder_names_and_paths") and MODEL_FOLDER_KEY not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[MODEL_FOLDER_KEY] = ([str(root)], set())


def _load_localization(locale: str = "zh-cn") -> dict[str, Any]:
    path = LOCAL_DIR / locale / "nodes.json"
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[Comfy-Easy-Qwen3-TTS] Failed to load localization file {path}: {exc}", flush=True)
        return {}
    return data if isinstance(data, dict) else {}


_LOCALIZATION = _load_localization(os.environ.get("COMFYUI_EASY_QWEN3_TTS_LOCALE", "zh-cn"))


def _tr(path: str, default: Any) -> Any:
    value: Any = _LOCALIZATION
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def tr_mapping(path: str, default: dict[str, str]) -> dict[str, str]:
    value = _tr(path, default)
    if isinstance(value, dict) and all(isinstance(key, str) and isinstance(val, str) for key, val in value.items()):
        return value
    return default


def ui(path: str, display_name: str, tooltip: str, **extra: Any) -> dict[str, Any]:
    value = _tr(f"ui.{path}", {})
    if isinstance(value, dict):
        display_name = value.get("display_name", display_name)
        tooltip = value.get("tooltip", tooltip)
    extra["display_name"] = display_name
    extra["tooltip"] = tooltip
    return extra


def _resolve_model_path(model_mode: str, model_path: str, allow_repo_id: bool) -> str:
    text = (model_path or "").strip()
    if not text or text.lower() == "auto":
        path = default_model_dir(model_mode)
        if path.is_dir():
            return str(path)
        if allow_repo_id:
            return DEFAULT_REPO_IDS[model_mode]
        return str(path)
    expanded = Path(text).expanduser()
    if expanded.is_absolute() or expanded.exists() or "/" not in text:
        return str(expanded)
    return text


def _resolve_tokenizer_dir(tokenizer_dir: str) -> Path | None:
    text = (tokenizer_dir or "").strip()
    if not text or text.lower() == "auto":
        path = default_tokenizer_dir()
        return path if path.is_dir() else None
    path = Path(text).expanduser()
    return path if path.is_dir() else None


def _has_config_json(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _repo_or_default(repo_id: str, default_repo_id: str) -> str:
    text = (repo_id or "").strip()
    return default_repo_id if not text or text.lower() == "auto" else text


def _endpoint_for_download(download_source: str, custom_endpoint_host: str) -> str | None:
    source = (download_source or "huggingface").strip().lower()
    if source == "huggingface":
        return None
    if source == "hf-mirror":
        return "https://hf-mirror.com"
    if source != "custom":
        raise ValueError(f"Unsupported download source: {download_source}")
    host = (custom_endpoint_host or "").strip()
    if not host:
        raise ValueError("Custom download source requires a host name.")
    if "://" not in host:
        host = f"https://{host}"
    parsed = urlparse(host)
    if not parsed.hostname:
        raise ValueError(f"Invalid custom endpoint host: {custom_endpoint_host}")
    return host.rstrip("/")


@contextmanager
def _patched_download_network(endpoint: str | None, custom_endpoint_ip: str, disable_ssl_verification: bool):
    parsed = urlparse(endpoint or "")
    override_host = parsed.hostname
    override_ip = (custom_endpoint_ip or "").strip()
    original_getaddrinfo = socket.getaddrinfo
    original_requests_request = None
    original_httpx_client_init = None
    original_httpx_async_client_init = None

    if override_host and override_ip:
        def patched_getaddrinfo(host, *args, **kwargs):
            if host == override_host:
                host = override_ip
            return original_getaddrinfo(host, *args, **kwargs)

        socket.getaddrinfo = patched_getaddrinfo

    if disable_ssl_verification:
        try:
            import requests

            original_requests_request = requests.sessions.Session.request

            def patched_requests_request(self, method, url, **kwargs):
                kwargs["verify"] = False
                return original_requests_request(self, method, url, **kwargs)

            requests.sessions.Session.request = patched_requests_request
        except Exception:
            original_requests_request = None

        try:
            import httpx

            original_httpx_client_init = httpx.Client.__init__
            original_httpx_async_client_init = httpx.AsyncClient.__init__

            def patched_httpx_client_init(self, *args, **kwargs):
                kwargs["verify"] = False
                return original_httpx_client_init(self, *args, **kwargs)

            def patched_httpx_async_client_init(self, *args, **kwargs):
                kwargs["verify"] = False
                return original_httpx_async_client_init(self, *args, **kwargs)

            httpx.Client.__init__ = patched_httpx_client_init
            httpx.AsyncClient.__init__ = patched_httpx_async_client_init
        except Exception:
            original_httpx_client_init = None
            original_httpx_async_client_init = None

    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo
        if original_requests_request is not None:
            import requests

            requests.sessions.Session.request = original_requests_request
        if original_httpx_client_init is not None and original_httpx_async_client_init is not None:
            import httpx

            httpx.Client.__init__ = original_httpx_client_init
            httpx.AsyncClient.__init__ = original_httpx_async_client_init


def _snapshot_download_to_dir(
    repo_id: str,
    local_dir: Path,
    download_source: str,
    custom_endpoint_host: str,
    custom_endpoint_ip: str,
    revision: str,
    disable_ssl_verification: bool,
    force_download: bool,
) -> str:
    endpoint = _endpoint_for_download(download_source, custom_endpoint_host)
    old_endpoint = os.environ.get("HF_ENDPOINT")
    old_disable_ssl = os.environ.get("HF_HUB_DISABLE_SSL_VERIFICATION")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    if disable_ssl_verification:
        os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        if old_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = old_endpoint
        if old_disable_ssl is None:
            os.environ.pop("HF_HUB_DISABLE_SSL_VERIFICATION", None)
        else:
            os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = old_disable_ssl
        raise ImportError("Online download requires huggingface_hub. Please install it in the ComfyUI Python environment.") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "force_download": bool(force_download),
    }
    if revision:
        kwargs["revision"] = revision

    signature = inspect.signature(snapshot_download)
    if "endpoint" in signature.parameters and endpoint:
        kwargs["endpoint"] = endpoint
    if "local_dir_use_symlinks" in signature.parameters:
        kwargs["local_dir_use_symlinks"] = False

    try:
        with _patched_download_network(endpoint, custom_endpoint_ip, bool(disable_ssl_verification)):
            return str(snapshot_download(**kwargs))
    finally:
        if old_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = old_endpoint
        if old_disable_ssl is None:
            os.environ.pop("HF_HUB_DISABLE_SSL_VERIFICATION", None)
        else:
            os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = old_disable_ssl


def _ensure_downloaded_snapshot(
    repo_id: str,
    local_dir: Path,
    download_source: str,
    custom_endpoint_host: str,
    custom_endpoint_ip: str,
    revision: str,
    disable_ssl_verification: bool,
    force_download: bool,
) -> str:
    if _has_config_json(local_dir) and not force_download:
        return f"exists: {local_dir}"
    downloaded = _snapshot_download_to_dir(
        repo_id=repo_id,
        local_dir=local_dir,
        download_source=download_source,
        custom_endpoint_host=custom_endpoint_host,
        custom_endpoint_ip=custom_endpoint_ip,
        revision=revision,
        disable_ssl_verification=disable_ssl_verification,
        force_download=force_download,
    )
    return f"downloaded: {repo_id} -> {downloaded}"


def _ensure_speech_tokenizer_link(model_path: str, tokenizer_dir: str, link_dependencies: bool) -> str:
    if not link_dependencies:
        return "disabled"
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return "model path is not a local directory"
    target = _resolve_tokenizer_dir(tokenizer_dir)
    if target is None:
        return "no external tokenizer directory"
    expected = model_dir / "speech_tokenizer"
    if (expected / "config.json").is_file():
        return "model directory already has speech_tokenizer"
    if not (target / "config.json").is_file():
        return f"{target} has no config.json"
    if expected.exists() or expected.is_symlink():
        return f"{expected} exists but is not a usable tokenizer"
    try:
        expected.symlink_to(target, target_is_directory=True)
        return f"linked {expected} -> {target}"
    except OSError:
        shutil.copytree(target, expected)
        return f"copied {target} -> {expected}"


def _dtype_from_precision(precision: str):
    value = (precision or "auto").strip().lower()
    if value == "auto":
        return None
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def _set_seed(seed: int) -> None:
    if int(seed) < 0:
        return
    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _generation_kwargs(
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    subtalker_dosample: bool,
    subtalker_top_k: int,
    subtalker_top_p: float,
    subtalker_temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "do_sample": bool(do_sample),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "temperature": float(temperature),
        "repetition_penalty": float(repetition_penalty),
        "subtalker_dosample": bool(subtalker_dosample),
        "subtalker_top_k": int(subtalker_top_k),
        "subtalker_top_p": float(subtalker_top_p),
        "subtalker_temperature": float(subtalker_temperature),
        "max_new_tokens": int(max_new_tokens),
    }


def _throw_if_interrupted() -> None:
    if comfy_model_management is not None and hasattr(comfy_model_management, "throw_exception_if_processing_interrupted"):
        comfy_model_management.throw_exception_if_processing_interrupted()


class _ComfyGenerationProgress:
    def __init__(self, max_new_tokens: int):
        self.total = max(1, int(max_new_tokens or 1))
        self.current = 0
        self.progress_bar = comfy_utils.ProgressBar(self.total) if comfy_utils is not None else None

    def begin(self) -> None:
        _throw_if_interrupted()
        if self.progress_bar is not None:
            self.progress_bar.update_absolute(0, self.total)

    def step(self) -> None:
        _throw_if_interrupted()
        if self.current < self.total:
            self.current += 1
        if self.progress_bar is not None:
            self.progress_bar.update_absolute(self.current, self.total)

    def finish(self) -> None:
        _throw_if_interrupted()
        if self.progress_bar is not None:
            self.progress_bar.update_absolute(self.total, self.total)


def _with_comfy_generation_progress(gen_kwargs: dict[str, Any]) -> tuple[_ComfyGenerationProgress, dict[str, Any]]:
    from qwen_tts.transformer_patch.transformers_4573.generation.stopping_criteria import (
        StoppingCriteria,
        StoppingCriteriaList,
    )

    class ComfyProgressStoppingCriteria(StoppingCriteria):
        def __init__(self, progress: _ComfyGenerationProgress):
            self.progress = progress

        def __call__(self, input_ids, scores, **kwargs):
            self.progress.step()
            return torch.full((input_ids.shape[0],), False, device=input_ids.device, dtype=torch.bool)

    progress = _ComfyGenerationProgress(int(gen_kwargs.get("max_new_tokens", 2048)))
    criteria = ComfyProgressStoppingCriteria(progress)
    updated = dict(gen_kwargs)
    existing = updated.get("stopping_criteria")
    if existing is None:
        updated["stopping_criteria"] = StoppingCriteriaList([criteria])
    else:
        if isinstance(existing, StoppingCriteriaList):
            combined = existing
        elif isinstance(existing, (list, tuple)):
            combined = StoppingCriteriaList(list(existing))
        else:
            combined = StoppingCriteriaList([existing])
        combined.append(criteria)
        updated["stopping_criteria"] = combined
    return progress, updated


def _audio_from_wavs(wavs: list[np.ndarray], sample_rate: int) -> dict[str, Any]:
    if not wavs:
        raise ValueError("Qwen3-TTS returned no audio.")
    tensors = []
    max_len = 0
    for wav in wavs:
        array = np.asarray(wav, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=-1)
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float().clamp(-1.0, 1.0).unsqueeze(0)
        tensors.append(tensor)
        max_len = max(max_len, int(tensor.shape[-1]))
    padded = []
    for tensor in tensors:
        if tensor.shape[-1] < max_len:
            tensor = torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[-1]))
        padded.append(tensor)
    return {"waveform": torch.stack(padded, dim=0), "sample_rate": int(sample_rate)}


def _select_comfy_audio(audio: dict[str, Any], batch_index: int) -> tuple[np.ndarray, int]:
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise TypeError("Expected ComfyUI AUDIO input with waveform and sample_rate.")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().float().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 3:
        raise ValueError(f"Expected AUDIO waveform [batch, channels, samples], got {tuple(waveform.shape)}.")
    index = max(0, min(int(batch_index), waveform.shape[0] - 1))
    mono = waveform[index].mean(dim=0).contiguous().numpy().astype(np.float32, copy=False)
    sample_rate = int(audio.get("sample_rate") or 44100)
    return mono, sample_rate


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class Qwen3TTSHandle:
    model: Any
    model_mode: str
    model_path: str
    precision: str
    device_map: str
    attn_implementation: str
    dependency_status: str

    def info(self) -> dict[str, Any]:
        return {
            "model_mode": self.model_mode,
            "model_path": self.model_path,
            "precision": self.precision,
            "device_map": self.device_map,
            "attn_implementation": self.attn_implementation,
            "dependency_status": self.dependency_status,
            "model_type": getattr(self.model.model, "tts_model_type", None),
            "model_size": getattr(self.model.model, "tts_model_size", None),
            "tokenizer_type": getattr(self.model.model, "tokenizer_type", None),
            "supported_languages": self.model.get_supported_languages(),
            "supported_speakers": self.model.get_supported_speakers(),
            "default_model_root": str(default_model_root()),
            "default_model_dirs": {key: str(default_model_dir(key)) for key in MODEL_MODE_OPTIONS},
            "default_speech_tokenizer_dir": str(default_tokenizer_dir()),
        }


def _load_qwen3_tts_handle(
    model_mode: str,
    resolved_model_path: str,
    speech_tokenizer_dir: str,
    device_map: str,
    precision: str,
    attn_implementation: str,
    dependency_status: str,
    low_cpu_mem_usage: bool,
    use_safetensors: bool,
    trust_remote_code: bool,
    reload_model: bool,
):
    _ensure_qwen_tts_on_path()
    dtype = _dtype_from_precision(precision)
    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "local_files_only": True,
        "low_cpu_mem_usage": bool(low_cpu_mem_usage),
        "use_safetensors": bool(use_safetensors),
        "trust_remote_code": bool(trust_remote_code),
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation
    key = (
        model_mode,
        resolved_model_path,
        str(_resolve_tokenizer_dir(speech_tokenizer_dir)),
        device_map,
        precision,
        attn_implementation,
        bool(low_cpu_mem_usage),
        bool(use_safetensors),
        bool(trust_remote_code),
    )
    with _RUNTIME_LOCK:
        if reload_model:
            _MODEL_CACHE.pop(key, None)
            _cleanup_memory()
        handle = _MODEL_CACHE.get(key)
        if handle is None:
            from qwen_tts import Qwen3TTSModel

            model = Qwen3TTSModel.from_pretrained(resolved_model_path, **kwargs)
            handle = Qwen3TTSHandle(
                model=model,
                model_mode=model_mode,
                model_path=resolved_model_path,
                precision=precision,
                device_map=device_map,
                attn_implementation=attn_implementation,
                dependency_status=dependency_status,
            )
            _MODEL_CACHE[key] = handle
    return handle


class ComfyEasyQwen3TTSLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_mode": (MODEL_MODE_OPTIONS, ui("load.model_mode", "模型模式", "选择要加载的 Qwen3-TTS 推理模型类型。")),
                "model_path": (
                    "STRING",
                    ui(
                        "load.model_path",
                        "模型路径",
                        "auto 使用 ComfyUI/models/qwen3-tts 下对应模式的子目录；也可填写本地绝对路径。",
                        default="auto",
                    ),
                ),
                "speech_tokenizer_dir": (
                    "STRING",
                    ui(
                        "load.speech_tokenizer_dir",
                        "语音 Tokenizer 目录",
                        "auto 使用 ComfyUI/models/qwen3-tts/speech-tokenizer。若主模型缺少 speech_tokenizer 子目录，会尝试创建链接。",
                        default="auto",
                    ),
                ),
                "device_map": (DEVICE_MAP_OPTIONS, ui("load.device_map", "计算设备", "传给 from_pretrained 的 device_map。")),
                "precision": (PRECISION_OPTIONS, ui("load.precision", "计算精度", "选择模型加载与推理计算精度。auto 使用 Transformers 默认行为。")),
                "attn_implementation": (ATTENTION_OPTIONS, ui("load.attn_implementation", "注意力实现", "auto 不显式传入；可选择 flash_attention_2、sdpa 或 eager。")),
                "link_dependencies": (
                    "BOOLEAN",
                    ui("load.link_dependencies", "链接依赖模型", "当主模型目录缺少 speech_tokenizer 时，将独立依赖目录链接或复制到主模型目录。", default=True),
                ),
                "low_cpu_mem_usage": ("BOOLEAN", ui("load.low_cpu_mem_usage", "低 CPU 内存加载", "传给 Transformers 的 low_cpu_mem_usage。", default=True)),
                "use_safetensors": ("BOOLEAN", ui("load.use_safetensors", "使用 Safetensors", "优先加载 safetensors 权重。", default=True)),
                "trust_remote_code": ("BOOLEAN", ui("load.trust_remote_code", "信任远程代码", "传给 Transformers 的 trust_remote_code。通常本插件不需要开启。", default=False)),
                "reload_model": ("BOOLEAN", ui("load.reload_model", "重新加载", "忽略插件缓存并重新实例化模型。", default=False)),
            },
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("模型", "信息")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "加载 Qwen3-TTS 模型，默认从 ComfyUI/models/qwen3-tts 下按模式读取主模型和依赖模型子目录。"

    def load(
        self,
        model_mode: str,
        model_path: str = "auto",
        speech_tokenizer_dir: str = "auto",
        device_map: str = "cuda:0",
        precision: str = "auto",
        attn_implementation: str = "auto",
        link_dependencies: bool = True,
        low_cpu_mem_usage: bool = True,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        reload_model: bool = False,
    ):
        model_mode = model_mode if model_mode in MODEL_MODE_OPTIONS else "custom-voice"
        resolved_model_path = _resolve_model_path(model_mode, model_path, False)
        dependency_status = _ensure_speech_tokenizer_link(resolved_model_path, speech_tokenizer_dir, bool(link_dependencies))
        handle = _load_qwen3_tts_handle(
            model_mode=model_mode,
            resolved_model_path=resolved_model_path,
            speech_tokenizer_dir=speech_tokenizer_dir,
            device_map=device_map,
            precision=precision,
            attn_implementation=attn_implementation,
            dependency_status=dependency_status,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            reload_model=reload_model,
        )
        return (handle, json.dumps(handle.info(), ensure_ascii=False, indent=2))


class ComfyEasyQwen3TTSOnlineLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_mode": (MODEL_MODE_OPTIONS, ui("online_load.model_mode", "模型模式", "选择要下载并加载的 Qwen3-TTS 推理模型类型。")),
                "download_source": (DOWNLOAD_SOURCE_OPTIONS, ui("online_load.download_source", "下载源", "选择 Hugging Face、hf-mirror 或自定义反向代理主机。")),
                "model_repo_id": (
                    "STRING",
                    ui(
                        "online_load.model_repo_id",
                        "主模型仓库",
                        "auto 使用所选模型模式对应的官方 Qwen 仓库；也可填写其他 Hugging Face 仓库 ID。",
                        default="auto",
                    ),
                ),
                "speech_tokenizer_repo_id": (
                    "STRING",
                    ui(
                        "online_load.speech_tokenizer_repo_id",
                        "语音 Tokenizer 仓库",
                        "auto 使用 Qwen/Qwen3-TTS-Tokenizer-12Hz；仅在本地缺少 tokenizer 依赖时下载。",
                        default="auto",
                    ),
                ),
                "custom_endpoint_host": (
                    "STRING",
                    ui(
                        "online_load.custom_endpoint_host",
                        "自定义 Host",
                        "下载源为 custom 时使用的反向代理主机名，例如 hf.example.com；未写协议时默认 https。",
                        default="",
                    ),
                ),
                "custom_endpoint_ip": (
                    "STRING",
                    ui(
                        "online_load.custom_endpoint_ip",
                        "自定义 IP",
                        "可选：将自定义 Host 解析到指定 IP 地址，保留 URL Host 用于反向代理和 TLS SNI。",
                        default="",
                    ),
                ),
                "model_revision": (
                    "STRING",
                    ui("online_load.model_revision", "主模型版本", "可选 Hugging Face revision、branch、tag 或 commit。", default=""),
                ),
                "speech_tokenizer_revision": (
                    "STRING",
                    ui("online_load.speech_tokenizer_revision", "Tokenizer 版本", "可选语音 Tokenizer 的 revision、branch、tag 或 commit。", default=""),
                ),
                "device_map": (DEVICE_MAP_OPTIONS, ui("online_load.device_map", "计算设备", "传给 from_pretrained 的 device_map。")),
                "precision": (PRECISION_OPTIONS, ui("online_load.precision", "计算精度", "选择模型加载与推理计算精度。auto 使用 Transformers 默认行为。")),
                "attn_implementation": (ATTENTION_OPTIONS, ui("online_load.attn_implementation", "注意力实现", "auto 不显式传入；可选择 flash_attention_2、sdpa 或 eager。")),
                "disable_ssl_verification": (
                    "BOOLEAN",
                    ui("online_load.disable_ssl_verification", "关闭 SSL 认证", "下载时为 requests 与 httpx 关闭证书校验。", default=False),
                ),
                "force_download": (
                    "BOOLEAN",
                    ui("online_load.force_download", "强制重新下载", "忽略本地 config.json 检查，重新调用 snapshot_download。", default=False),
                ),
                "low_cpu_mem_usage": ("BOOLEAN", ui("online_load.low_cpu_mem_usage", "低 CPU 内存加载", "传给 Transformers 的 low_cpu_mem_usage。", default=True)),
                "use_safetensors": ("BOOLEAN", ui("online_load.use_safetensors", "使用 Safetensors", "优先加载 safetensors 权重。", default=True)),
                "trust_remote_code": ("BOOLEAN", ui("online_load.trust_remote_code", "信任远程代码", "传给 Transformers 的 trust_remote_code。通常本插件不需要开启。", default=False)),
                "reload_model": ("BOOLEAN", ui("online_load.reload_model", "重新加载", "忽略插件缓存并重新实例化模型。", default=False)),
            },
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("模型", "信息")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "检查本地模型目录，缺失时从 Hugging Face、hf-mirror 或自定义反向代理下载主模型和依赖模型，下载完成后加载。"

    def load(
        self,
        model_mode: str,
        download_source: str = "huggingface",
        model_repo_id: str = "auto",
        speech_tokenizer_repo_id: str = "auto",
        custom_endpoint_host: str = "",
        custom_endpoint_ip: str = "",
        model_revision: str = "",
        speech_tokenizer_revision: str = "",
        device_map: str = "cuda:0",
        precision: str = "auto",
        attn_implementation: str = "auto",
        disable_ssl_verification: bool = False,
        force_download: bool = False,
        low_cpu_mem_usage: bool = True,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        reload_model: bool = False,
    ):
        model_mode = model_mode if model_mode in MODEL_MODE_OPTIONS else "custom-voice"
        local_model_dir = default_model_dir(model_mode)
        local_tokenizer_dir = default_tokenizer_dir()
        resolved_model_repo_id = _repo_or_default(model_repo_id, DEFAULT_REPO_IDS[model_mode])
        resolved_tokenizer_repo_id = _repo_or_default(speech_tokenizer_repo_id, DEFAULT_TOKENIZER_REPO_ID)

        model_status = _ensure_downloaded_snapshot(
            repo_id=resolved_model_repo_id,
            local_dir=local_model_dir,
            download_source=download_source,
            custom_endpoint_host=custom_endpoint_host,
            custom_endpoint_ip=custom_endpoint_ip,
            revision=(model_revision or "").strip(),
            disable_ssl_verification=bool(disable_ssl_verification),
            force_download=bool(force_download),
        )

        embedded_tokenizer_dir = local_model_dir / "speech_tokenizer"
        if _has_config_json(embedded_tokenizer_dir):
            tokenizer_status = f"exists: {embedded_tokenizer_dir}"
        else:
            tokenizer_status = _ensure_downloaded_snapshot(
                repo_id=resolved_tokenizer_repo_id,
                local_dir=local_tokenizer_dir,
                download_source=download_source,
                custom_endpoint_host=custom_endpoint_host,
                custom_endpoint_ip=custom_endpoint_ip,
                revision=(speech_tokenizer_revision or "").strip(),
                disable_ssl_verification=bool(disable_ssl_verification),
                force_download=bool(force_download),
            )

        link_status = _ensure_speech_tokenizer_link(str(local_model_dir), str(local_tokenizer_dir), True)
        dependency_status = "; ".join([model_status, tokenizer_status, link_status])
        handle = _load_qwen3_tts_handle(
            model_mode=model_mode,
            resolved_model_path=str(local_model_dir),
            speech_tokenizer_dir=str(local_tokenizer_dir),
            device_map=device_map,
            precision=precision,
            attn_implementation=attn_implementation,
            dependency_status=dependency_status,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            reload_model=reload_model,
        )
        return (handle, json.dumps(handle.info(), ensure_ascii=False, indent=2))


def common_generation_inputs() -> dict[str, Any]:
    return {
        "seed": ("INT", ui("generation.seed", "随机种子", "-1 表示每次随机；非负数会固定 numpy/torch 随机种子。", default=-1, min=-1, max=0x7FFFFFFF, step=1)),
        "do_sample": ("BOOLEAN", ui("generation.do_sample", "启用采样", "开启采样生成。", default=True)),
        "top_k": ("INT", ui("generation.top_k", "Top K", "Top-k 采样参数。", default=50, min=0, max=4096, step=1)),
        "top_p": ("FLOAT", ui("generation.top_p", "Top P", "Top-p 采样参数。", default=1.0, min=0.01, max=1.0, step=0.01)),
        "temperature": ("FLOAT", ui("generation.temperature", "温度", "采样温度，越高越随机。", default=0.9, min=0.01, max=5.0, step=0.01)),
        "repetition_penalty": ("FLOAT", ui("generation.repetition_penalty", "重复惩罚", "降低重复 token/code 的惩罚系数。", default=1.05, min=0.0, max=10.0, step=0.01)),
        "subtalker_dosample": ("BOOLEAN", ui("generation.subtalker_dosample", "Subtalker 采样", "启用 subtalker 采样。", default=True)),
        "subtalker_top_k": ("INT", ui("generation.subtalker_top_k", "Subtalker Top K", "Subtalker top-k 采样参数。", default=50, min=0, max=4096, step=1)),
        "subtalker_top_p": ("FLOAT", ui("generation.subtalker_top_p", "Subtalker Top P", "Subtalker top-p 采样参数。", default=1.0, min=0.01, max=1.0, step=0.01)),
        "subtalker_temperature": ("FLOAT", ui("generation.subtalker_temperature", "Subtalker 温度", "Subtalker 采样温度。", default=0.9, min=0.01, max=5.0, step=0.01)),
        "max_new_tokens": ("INT", ui("generation.max_new_tokens", "最大新 Token", "生成的最大 codec token 数量。", default=2048, min=1, max=16384, step=1)),
        "unload_model_after_generation": ("BOOLEAN", ui("generation.unload_model_after_generation", "生成后清理显存", "生成后调用 gc 与 CUDA cache 清理；不会移除加载节点输出对象。", default=False)),
    }


def _voice_clone_reference_mode(ref_text: str, x_vector_only_mode: bool) -> tuple[str | None, bool, bool]:
    normalized_ref_text = (ref_text or "").strip()
    requested_x_vector_only = bool(x_vector_only_mode)
    if normalized_ref_text:
        return normalized_ref_text, requested_x_vector_only, False
    if not requested_x_vector_only:
        print(
            "[Comfy-Easy-Qwen3-TTS] ref_text is empty; using x_vector_only_mode=True for voice clone.",
            flush=True,
        )
    return None, True, not requested_x_vector_only


class _Qwen3TTSInferBase:
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("音频", "信息")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if int(kwargs.get("seed", -1)) < 0:
            return random.random()
        return None

    def _finish(
        self,
        handle: Qwen3TTSHandle,
        wavs: list[np.ndarray],
        sample_rate: int,
        unload_model_after_generation: bool,
        extra_info: dict[str, Any] | None = None,
    ):
        audio = _audio_from_wavs(wavs, sample_rate)
        info = {
            "sample_rate": int(sample_rate),
            "samples": int(audio["waveform"].shape[-1]),
            "duration_seconds": int(audio["waveform"].shape[-1]) / float(sample_rate),
            "model_mode": handle.model_mode,
            "model_path": handle.model_path,
        }
        if extra_info:
            info.update(extra_info)
        if unload_model_after_generation:
            _cleanup_memory()
        return (audio, json.dumps(info, ensure_ascii=False, indent=2))


class ComfyEasyQwen3TTSCustomVoice(_Qwen3TTSInferBase):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "qwen3_tts_model": (MODEL_TYPE, ui("custom_voice.qwen3_tts_model", "Qwen3-TTS 模型", "加载模型节点输出的 Custom Voice 模型。")),
            "text": ("STRING", ui("custom_voice.text", "合成文本", "需要合成的文本。", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。", multiline=True)),
            "language": ("STRING", ui("custom_voice.language", "语言", "语言名称，例如 Chinese、English；Auto 由模型自动处理。", default="Chinese")),
            "speaker": ("STRING", ui("custom_voice.speaker", "说话人", "Custom Voice 模型支持的说话人名称，例如 Vivian。", default="Vivian")),
            "instruct": ("STRING", ui("custom_voice.instruct", "风格指令", "可选风格或情绪指令。", default="", multiline=True)),
            "non_streaming_mode": ("BOOLEAN", ui("custom_voice.non_streaming_mode", "非流式文本", "使用非流式文本输入。", default=True)),
        }
        required.update(common_generation_inputs())
        return {"required": required}

    DESCRIPTION = "使用 Qwen3-TTS Custom Voice 模型按预置说话人合成语音。"

    def generate(self, qwen3_tts_model: Qwen3TTSHandle, text: str, language: str, speaker: str, instruct: str, non_streaming_mode: bool, **kwargs):
        _set_seed(int(kwargs.pop("seed", -1)))
        unload = bool(kwargs.pop("unload_model_after_generation", False))
        progress, gen_kwargs = _with_comfy_generation_progress(_generation_kwargs(**kwargs))
        progress.begin()
        wavs, sample_rate = qwen3_tts_model.model.generate_custom_voice(
            text=text,
            language=language or "Auto",
            speaker=speaker,
            instruct=instruct or "",
            non_streaming_mode=bool(non_streaming_mode),
            **gen_kwargs,
        )
        progress.finish()
        return self._finish(qwen3_tts_model, wavs, sample_rate, unload)


class ComfyEasyQwen3TTSVoiceDesign(_Qwen3TTSInferBase):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "qwen3_tts_model": (MODEL_TYPE, ui("voice_design.qwen3_tts_model", "Qwen3-TTS 模型", "加载模型节点输出的 Voice Design 模型。")),
            "text": ("STRING", ui("voice_design.text", "合成文本", "需要合成的文本。", default="哥哥，你回来啦，人家等了你好久好久了，要抱抱！", multiline=True)),
            "language": ("STRING", ui("voice_design.language", "语言", "语言名称，例如 Chinese、English；Auto 由模型自动处理。", default="Chinese")),
            "instruct": ("STRING", ui("voice_design.instruct", "音色设计指令", "用自然语言描述希望生成的音色、年龄、语气或风格。", default="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。", multiline=True)),
            "non_streaming_mode": ("BOOLEAN", ui("voice_design.non_streaming_mode", "非流式文本", "使用非流式文本输入。", default=True)),
        }
        required.update(common_generation_inputs())
        return {"required": required}

    DESCRIPTION = "使用 Qwen3-TTS Voice Design 模型按自然语言音色描述合成语音。"

    def generate(self, qwen3_tts_model: Qwen3TTSHandle, text: str, language: str, instruct: str, non_streaming_mode: bool, **kwargs):
        _set_seed(int(kwargs.pop("seed", -1)))
        unload = bool(kwargs.pop("unload_model_after_generation", False))
        progress, gen_kwargs = _with_comfy_generation_progress(_generation_kwargs(**kwargs))
        progress.begin()
        wavs, sample_rate = qwen3_tts_model.model.generate_voice_design(
            text=text,
            language=language or "Auto",
            instruct=instruct or "",
            non_streaming_mode=bool(non_streaming_mode),
            **gen_kwargs,
        )
        progress.finish()
        return self._finish(qwen3_tts_model, wavs, sample_rate, unload)


class ComfyEasyQwen3TTSVoiceClone(_Qwen3TTSInferBase):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "qwen3_tts_model": (MODEL_TYPE, ui("voice_clone.qwen3_tts_model", "Qwen3-TTS 模型", "加载模型节点输出的 Voice Clone/Base 模型。")),
            "ref_audio": ("AUDIO", ui("voice_clone.ref_audio", "参考音频", "用于克隆音色的 ComfyUI AUDIO 输入。")),
            "text": ("STRING", ui("voice_clone.text", "合成文本", "需要合成的文本。", default="She said she would be here by noon.", multiline=True)),
            "language": ("STRING", ui("voice_clone.language", "语言", "语言名称，例如 Chinese、English；Auto 由模型自动处理。", default="English")),
            "ref_text": ("STRING", ui("voice_clone.ref_text", "参考文本", "参考音频对应文本。留空时自动使用仅说话人向量模式。", default="", multiline=True)),
            "x_vector_only_mode": ("BOOLEAN", ui("voice_clone.x_vector_only_mode", "仅说话人向量", "只使用说话人向量克隆音色；关闭后会使用参考文本进入 ICL 模式。", default=True)),
            "ref_audio_batch_index": ("INT", ui("voice_clone.ref_audio_batch_index", "参考音频批次", "当 AUDIO 含多个 batch 时选择使用哪一个。", default=0, min=0, max=4096, step=1)),
            "non_streaming_mode": ("BOOLEAN", ui("voice_clone.non_streaming_mode", "非流式文本", "使用非流式文本输入。", default=False)),
        }
        required.update(common_generation_inputs())
        return {"required": required}

    DESCRIPTION = "使用 Qwen3-TTS Base 模型和参考音频进行语音克隆合成。"

    def generate(
        self,
        qwen3_tts_model: Qwen3TTSHandle,
        ref_audio,
        text: str,
        language: str,
        ref_text: str,
        x_vector_only_mode: bool,
        ref_audio_batch_index: int,
        non_streaming_mode: bool,
        **kwargs,
    ):
        _set_seed(int(kwargs.pop("seed", -1)))
        unload = bool(kwargs.pop("unload_model_after_generation", False))
        ref_waveform, ref_sample_rate = _select_comfy_audio(ref_audio, int(ref_audio_batch_index))
        effective_ref_text, effective_x_vector_only_mode, auto_x_vector_only_mode = _voice_clone_reference_mode(ref_text, x_vector_only_mode)
        progress, gen_kwargs = _with_comfy_generation_progress(_generation_kwargs(**kwargs))
        progress.begin()
        wavs, sample_rate = qwen3_tts_model.model.generate_voice_clone(
            text=text,
            language=language or "Auto",
            ref_audio=(ref_waveform, ref_sample_rate),
            ref_text=effective_ref_text,
            x_vector_only_mode=effective_x_vector_only_mode,
            non_streaming_mode=bool(non_streaming_mode),
            **gen_kwargs,
        )
        progress.finish()
        return self._finish(
            qwen3_tts_model,
            wavs,
            sample_rate,
            unload,
            {
                "voice_clone_mode": "x_vector_only" if effective_x_vector_only_mode else "icl",
                "x_vector_only_mode": effective_x_vector_only_mode,
                "ref_text_provided": effective_ref_text is not None,
                "auto_x_vector_only_mode": auto_x_vector_only_mode,
            },
        )


NODE_CLASS_MAPPINGS = {
    "ComfyEasyQwen3TTSLoadModel": ComfyEasyQwen3TTSLoadModel,
    "ComfyEasyQwen3TTSOnlineLoadModel": ComfyEasyQwen3TTSOnlineLoadModel,
    "ComfyEasyQwen3TTSCustomVoice": ComfyEasyQwen3TTSCustomVoice,
    "ComfyEasyQwen3TTSVoiceDesign": ComfyEasyQwen3TTSVoiceDesign,
    "ComfyEasyQwen3TTSVoiceClone": ComfyEasyQwen3TTSVoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = tr_mapping(
    "node_display_names",
    {
        "ComfyEasyQwen3TTSLoadModel": "Qwen3-TTS Load Model",
        "ComfyEasyQwen3TTSOnlineLoadModel": "Qwen3-TTS Online Load Model",
        "ComfyEasyQwen3TTSCustomVoice": "Qwen3-TTS Custom Voice",
        "ComfyEasyQwen3TTSVoiceDesign": "Qwen3-TTS Voice Design",
        "ComfyEasyQwen3TTSVoiceClone": "Qwen3-TTS Voice Clone",
    },
)
