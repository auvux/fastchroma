"""Fast constant-Q transform and chromagram.

Transform objects build the filterbank once and reuse it across calls — ideal
for batch / dataset preprocessing:

    import fastchroma, numpy as np
    y = np.random.randn(22050 * 30).astype(np.float32)

    cqt = fastchroma.CQT(sr=22050, hop_length=512, n_bins=84)
    mag = cqt(y)                      # (n_bins, n_frames) magnitude
    z   = cqt(y, output="complex")    # complex64, phase preserved
    f   = cqt.frequencies             # bin centre frequencies (Hz)

    chroma = fastchroma.Chroma(sr=22050, hop_length=512)(y)   # (12, n_frames)

One-shot functional forms (``cqt``, ``chroma``) are also provided. Mono input,
equal temperament.

On Apple-silicon Macs the transform runs on the GPU (Metal) by default, and on
NVIDIA GPUs via CUDA — all output variants, including complex. Pass
``backend="cpu"`` / ``backend="gpu"`` (or the explicit ``"metal"`` /
``"cuda"``) to force a path, or set the ``FASTCHROMA_BACKEND`` environment
variable.

Output mirrors input (like fastmel): numpy in, numpy out; a torch tensor in,
a torch tensor out on the same device. On CUDA builds, torch CUDA tensors and
CuPy arrays are processed in place — no host round trip — via DLPack, with
the work enqueued on the caller's current stream. torch MPS tensors are
staged through unified memory and computed by the Metal backend. torch/CuPy
are imported lazily, never required.
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np

from . import _fastchroma
from ._version import __version__

__all__ = ["CQT", "Chroma", "cqt", "chroma", "cqt_frequencies", "gpu_available",
           "gpu_backend", "metal_available", "__version__"]

_C1_HZ = 32.70319566257483  # note C1 = 440 * 2 ** ((24 - 69) / 12), A440 / 12-TET

_BACKENDS = ("auto", "cpu", "gpu", "metal", "cuda")
_OUTPUTS = ("magnitude", "complex", "power", "db")
_GPU_MODE = {"magnitude": 0, "power": 1, "db": 2}


def gpu_available():
    """True when this build's GPU backend (Metal or CUDA) can run here."""
    return _fastchroma.gpu_available()


def gpu_backend():
    """Which GPU backend this build carries: 'metal', 'cuda', or 'none'."""
    return _fastchroma.gpu_backend()


def metal_available():
    """True when the Metal (GPU) backend can run on this machine."""
    return _fastchroma.gpu_backend() == "metal" and _fastchroma.gpu_available()


def _as_mono_f32(y):
    """Normalize input to a float32 numpy array.

    Returns (array, rewrap) where rewrap converts a numpy result back to the
    caller's library, mirroring fastmel: numpy in -> numpy out, torch in ->
    torch out (same device). torch is imported lazily, never required.

    torch MPS tensors are staged through host memory (cheap on unified
    memory) and the result returns on MPS. torch CUDA tensors and CuPy
    arrays normally take the resident DLPack path before reaching here;
    this host staging is their fallback (CPU build, or parameters the GPU
    path does not support), and the result still returns on their device.
    """
    module = type(y).__module__.partition(".")[0]
    if module == "torch":
        import torch
        device = y.device
        if device.type not in ("cpu", "mps", "cuda"):
            raise TypeError(
                f"fastchroma has no {device.type} backend; move the tensor to the "
                "host first (y.cpu())")
        t = y.detach()
        if device.type != "cpu":
            t = t.cpu()
        arr = np.ascontiguousarray(t.numpy(), dtype=np.float32).ravel()
        if device.type != "cpu":
            return arr, lambda out: torch.from_numpy(out).to(device)
        return arr, torch.from_numpy
    if module == "cupy":
        import cupy
        arr = np.ascontiguousarray(cupy.asnumpy(y), dtype=np.float32).ravel()
        return arr, cupy.asarray
    return np.ascontiguousarray(y, dtype=np.float32).ravel(), lambda out: out


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        backend = os.environ.get("FASTCHROMA_BACKEND", "auto").lower() or "auto"
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}; got {backend!r}")
    if backend in ("metal", "cuda") and _fastchroma.gpu_backend() != backend:
        raise RuntimeError(
            f"this fastchroma build has no {backend} backend "
            f"(its GPU backend is {_fastchroma.gpu_backend()!r})")
    return backend


def _gpu_unavailable_error(backend: str) -> RuntimeError:
    return RuntimeError(
        f"{backend} backend unavailable on this machine or unsupported for "
        "these parameters (e.g. a hop without enough factors of two)")


_NOT_RESIDENT = object()


def _try_resident(y, call, forced):
    """GPU-resident path: DLPack in, DLPack out, same device (as in fastmel).

    `call(capsule, stream)` runs the C++ resident compute and returns an
    output capsule. The kernels are enqueued on the caller's current CUDA
    stream (DLPack stream handoff) and the call returns without host
    synchronization, like a native framework op; the input is made visible
    to that stream by ``__dlpack__(stream=...)``.

    Returns _NOT_RESIDENT when the input is not a GPU-resident tensor or
    this build cannot take it; the caller falls back to host staging. A
    RuntimeError (e.g. an unsupported hop) likewise falls back unless the
    backend was forced.
    """
    if _fastchroma.gpu_backend() != "cuda":
        return _NOT_RESIDENT
    module = type(y).__module__.partition(".")[0]
    if module == "torch":
        import torch
        if y.device.type != "cuda":
            return _NOT_RESIDENT
        if y.device.index not in (None, 0):
            raise NotImplementedError("only CUDA device 0 is supported")
        src = y.detach()
        if src.dtype != torch.float32:
            src = src.float()
        src = src.reshape(-1).contiguous()
        stream = torch.cuda.current_stream().cuda_stream or 1  # 1 = legacy default
    elif module == "cupy":
        import cupy
        src = y
        if src.dtype != cupy.float32:
            src = src.astype(cupy.float32)
        src = cupy.ascontiguousarray(src).ravel()
        stream = cupy.cuda.get_current_stream().ptr or 1
    else:
        return _NOT_RESIDENT
    try:
        out = call(src.__dlpack__(stream=stream), stream)
    except RuntimeError:
        if forced:
            raise
        return _NOT_RESIDENT
    if module == "torch":
        import torch
        return torch.from_dlpack(out)
    import cupy
    try:
        return cupy.from_dlpack(out)
    except TypeError:  # older CuPy takes capsules via fromDlpack
        return cupy.fromDlpack(out)


def cqt_frequencies(n_bins, *, fmin=0.0, bins_per_octave=12):
    """Centre frequencies (Hz) of the CQT bins. fmin=0 selects C1."""
    f0 = fmin if fmin > 0 else _C1_HZ
    return f0 * 2.0 ** (np.arange(int(n_bins)) / bins_per_octave)


def _warn_if_odd_hop(hop_length: int, n_octaves: int) -> None:
    step = 1 << (n_octaves - 1)
    if hop_length % step:
        warnings.warn(
            f"hop_length={hop_length} is not divisible by 2**(n_octaves-1)={step}; "
            "lower octaves run at full rate (slower, more memory). Prefer 512 or 1024.",
            stacklevel=3,
        )


class CQT:
    """Constant-Q transform with a reusable filterbank.

    fmin=0 selects C1 (32.7 Hz). Use a hop divisible by ``2**(n_octaves-1)``.
    """

    def __init__(self, *, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12, fmin=0.0):
        self.sr = float(sr)
        self.hop_length = int(hop_length)
        self.n_bins = int(n_bins)
        self.bins_per_octave = int(bins_per_octave)
        self.fmin = float(fmin)
        n_octaves = math.ceil(self.n_bins / self.bins_per_octave)
        _warn_if_odd_hop(self.hop_length, n_octaves)
        self.frequencies = cqt_frequencies(self.n_bins, fmin=self.fmin,
                                            bins_per_octave=self.bins_per_octave)

    def __repr__(self):
        return (f"CQT(sr={self.sr:g}, hop_length={self.hop_length}, n_bins={self.n_bins}, "
                f"bins_per_octave={self.bins_per_octave}, fmin={self.fmin:g})")

    def __call__(self, y, *, output="magnitude", backend="auto"):
        if output not in _OUTPUTS:
            raise ValueError(f"output must be one of {_OUTPUTS}; got {output!r}")
        backend = _resolve_backend(backend)

        if backend != "cpu":
            mode = 3 if output == "complex" else _GPU_MODE[output]
            out = _try_resident(
                y, lambda cap, stream: _fastchroma.cqt_dlpack(
                    cap, self.sr, self.hop_length, self.fmin, self.n_bins,
                    self.bins_per_octave, mode, stream),
                forced=backend != "auto")
            if out is not _NOT_RESIDENT:
                return out

        y, rewrap = _as_mono_f32(y)
        args = (y, self.sr, self.hop_length, self.fmin, self.n_bins, self.bins_per_octave)

        if backend != "cpu":
            if output == "complex":
                m = _fastchroma.cqt_complex_gpu(*args)
            else:
                m = _fastchroma.cqt_gpu(*args, _GPU_MODE[output])  # in-kernel epilogue
            if m is not None:
                return rewrap(m)
            if backend != "auto":
                raise _gpu_unavailable_error(backend)

        if output == "complex":
            return rewrap(_fastchroma.cqt_complex(*args))
        m = _fastchroma.cqt(*args)
        if output == "power":
            m = m * m
        elif output == "db":
            m = 10.0 * np.log10(np.maximum(m * m, 1e-10), dtype=np.float32)
        return rewrap(m)


class Chroma:
    """CQT chromagram with a reusable filterbank. fmin=0 selects C1."""

    def __init__(self, *, sr=22050, hop_length=512, bins_per_octave=36, n_octaves=7,
                 n_chroma=12, fmin=0.0):
        self.sr = float(sr)
        self.hop_length = int(hop_length)
        self.bins_per_octave = int(bins_per_octave)
        self.n_octaves = int(n_octaves)
        self.n_chroma = int(n_chroma)
        self.fmin = float(fmin)
        _warn_if_odd_hop(self.hop_length, self.n_octaves)

    def __repr__(self):
        return (f"Chroma(sr={self.sr:g}, hop_length={self.hop_length}, "
                f"bins_per_octave={self.bins_per_octave}, n_octaves={self.n_octaves}, "
                f"n_chroma={self.n_chroma}, fmin={self.fmin:g})")

    def __call__(self, y, *, backend="auto"):
        backend = _resolve_backend(backend)

        if backend != "cpu":
            out = _try_resident(
                y, lambda cap, stream: _fastchroma.chroma_dlpack(
                    cap, self.sr, self.hop_length, self.bins_per_octave,
                    self.n_octaves, self.n_chroma, self.fmin, stream),
                forced=backend != "auto")
            if out is not _NOT_RESIDENT:
                return out

        y, rewrap = _as_mono_f32(y)
        args = (y, self.sr, self.hop_length, self.bins_per_octave, self.n_octaves,
                self.n_chroma, self.fmin)
        if backend != "cpu":
            m = _fastchroma.chroma_gpu(*args)
            if m is not None:
                return rewrap(m)
            if backend != "auto":
                raise _gpu_unavailable_error(backend)
        return rewrap(_fastchroma.chroma(*args))


def cqt(y, *, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12, fmin=0.0,
        output="magnitude", backend="auto"):
    """One-shot CQT, shape (n_bins, n_frames).

    output: 'magnitude' (default, float32), 'complex' (complex64, phase
    preserved), 'power' (float32), or 'db' (float32,
    10*log10(max(power, 1e-10)), as in fastmel).
    """
    return CQT(sr=sr, hop_length=hop_length, n_bins=n_bins,
               bins_per_octave=bins_per_octave, fmin=fmin)(y, output=output, backend=backend)


def chroma(y, *, sr=22050, hop_length=512, bins_per_octave=36, n_octaves=7,
           n_chroma=12, fmin=0.0, backend="auto"):
    """One-shot CQT chromagram, shape (n_chroma, n_frames)."""
    return Chroma(sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave,
                  n_octaves=n_octaves, n_chroma=n_chroma, fmin=fmin)(y, backend=backend)
