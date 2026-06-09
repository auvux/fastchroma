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

One-shot functional forms (``cqt_magnitude``, ``cqt_complex``, ``chroma_cqt``)
are also provided. Mono input, equal temperament.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from . import _fastchroma
from ._version import __version__

__all__ = ["CQT", "Chroma", "cqt_magnitude", "cqt_complex", "chroma_cqt",
           "cqt_frequencies", "__version__"]

_C1_HZ = 32.70319566257483  # note C1 = 440 * 2 ** ((24 - 69) / 12), A440 / 12-TET


def _as_mono_f32(y) -> np.ndarray:
    return np.ascontiguousarray(y, dtype=np.float32).ravel()


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

    def __call__(self, y, *, output="magnitude"):
        y = _as_mono_f32(y)
        args = (y, self.sr, self.hop_length, self.fmin, self.n_bins, self.bins_per_octave)
        if output == "magnitude":
            return _fastchroma.cqt_magnitude(*args)
        if output == "complex":
            return _fastchroma.cqt_complex(*args)
        if output == "power":
            m = _fastchroma.cqt_magnitude(*args)
            return m * m
        raise ValueError(f"output must be 'magnitude', 'complex', or 'power'; got {output!r}")


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

    def __call__(self, y):
        y = _as_mono_f32(y)
        return _fastchroma.chroma_cqt(y, self.sr, self.hop_length, self.bins_per_octave,
                                      self.n_octaves, self.n_chroma, self.fmin)


def cqt_magnitude(y, *, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12, fmin=0.0):
    """One-shot magnitude CQT, shape (n_bins, n_frames)."""
    return CQT(sr=sr, hop_length=hop_length, n_bins=n_bins,
               bins_per_octave=bins_per_octave, fmin=fmin)(y)


def cqt_complex(y, *, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12, fmin=0.0):
    """One-shot complex CQT (complex64), shape (n_bins, n_frames)."""
    return CQT(sr=sr, hop_length=hop_length, n_bins=n_bins,
               bins_per_octave=bins_per_octave, fmin=fmin)(y, output="complex")


def chroma_cqt(y, *, sr=22050, hop_length=512, bins_per_octave=36, n_octaves=7,
               n_chroma=12, fmin=0.0):
    """One-shot CQT chromagram, shape (n_chroma, n_frames)."""
    return Chroma(sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave,
                  n_octaves=n_octaves, n_chroma=n_chroma, fmin=fmin)(y)
