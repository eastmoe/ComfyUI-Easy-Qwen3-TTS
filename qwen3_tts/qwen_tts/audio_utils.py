# coding=utf-8
"""Audio helpers backed by torch/torchaudio."""

from typing import BinaryIO, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF


AudioSource = Union[str, BinaryIO]


def _to_mono_float32_np(waveform: torch.Tensor) -> np.ndarray:
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.dim() != 1:
        raise RuntimeError(f"Expected 1-D or 2-D audio tensor, got shape={tuple(waveform.shape)}")
    return waveform.detach().cpu().to(torch.float32).numpy().astype(np.float32, copy=False)


def load_audio_mono_float32(source: AudioSource) -> Tuple[np.ndarray, int]:
    """Load audio with torchaudio and return mono float32 numpy waveform plus sample rate."""
    try:
        waveform, sample_rate = torchaudio.load(source, channels_first=True)
    except Exception as exc:
        raise RuntimeError(
            "Unsupported audio format or failed to load audio with torchaudio. "
            "Please provide a format supported by the active torchaudio backend."
        ) from exc
    return _to_mono_float32_np(waveform), int(sample_rate)


def resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a numpy waveform with torchaudio.functional.resample."""
    if int(orig_sr) == int(target_sr):
        return audio.astype(np.float32, copy=False)

    tensor = torch.as_tensor(audio, dtype=torch.float32)
    if tensor.dim() > 1:
        tensor = tensor.mean(dim=-1)
    resampled = AF.resample(tensor, int(orig_sr), int(target_sr))
    return resampled.detach().cpu().numpy().astype(np.float32, copy=False)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: Optional[float],
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Return a Slaney mel filterbank shaped as (n_mels, n_freqs)."""
    if f_max is None:
        f_max = float(sample_rate) / 2.0
    fbanks = AF.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=float(f_min),
        f_max=float(f_max),
        n_mels=int(n_mels),
        sample_rate=int(sample_rate),
        norm="slaney",
        mel_scale="slaney",
    )
    fbanks = fbanks.transpose(0, 1).contiguous().to(torch.float32)
    if device is not None:
        fbanks = fbanks.to(device)
    return fbanks


def torch_norm_db(audio: torch.Tensor, db_level: float = -6.0) -> torch.Tensor:
    """Peak-normalize audio to the requested dBFS level using torch."""
    target_peak = 10.0 ** (float(db_level) / 20.0)
    peak = audio.abs().max()
    if peak <= 0:
        return audio
    return audio * (target_peak / peak)
