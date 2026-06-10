# fastchroma

Fast constant-Q transform and chromagram for Python, with a small C++ core.

The constant-Q transform (Brown 1991; Schörkhuber & Klapuri 2010) and the
chromagram (Fujishima 1999) are standard music-analysis features. `fastchroma`
implements them directly in C++ with a per-platform FFT backend and a recursive
octave decimation that keeps memory bounded. On Apple-silicon Macs the whole
transform additionally runs on the GPU (Metal), and on NVIDIA GPUs (CUDA). Its
only runtime dependency is NumPy (inputs and outputs are NumPy arrays).

## Install

```bash
pip install fastchroma
```

Wheels ship for macOS, Linux, and Windows. Building from source needs a C++17
compiler and CMake.

## Usage

Transform objects build the filterbank once and reuse it — best for batch /
dataset preprocessing:

```python
import numpy as np
import fastchroma

y = np.random.randn(22050 * 30).astype(np.float32)

cqt = fastchroma.CQT(sr=22050, hop_length=512, n_bins=84)
mag = cqt(y)                    # (n_bins, n_frames) magnitude
z   = cqt(y, output="complex")  # complex64, phase preserved
f   = cqt.frequencies           # bin centre frequencies (Hz)

chroma = fastchroma.Chroma(sr=22050, hop_length=512)(y)   # (12, n_frames)
```

One-shot functional forms exist too; output variants are selected with the
`output=` kwarg:

```python
mag    = fastchroma.cqt(y, sr=22050, n_bins=84)          # magnitude (default) — float32
z      = fastchroma.cqt(y, sr=22050, output="complex")   # complex64, phase preserved
power  = fastchroma.cqt(y, sr=22050, output="power")     # float32
db     = fastchroma.cqt(y, sr=22050, output="db")        # float32, 10*log10(max(power, 1e-10))
chroma = fastchroma.chroma(y, sr=22050)
```

Output mirrors input (as in fastmel): numpy in → numpy out; a torch tensor in
→ a torch tensor out on the same device. On CUDA builds, torch CUDA tensors
and CuPy arrays are processed in place — no host round trip — via DLPack,
with the kernels enqueued on the caller's current stream like a native
framework op. torch MPS tensors are staged through unified memory and
computed by the Metal backend. torch/CuPy are imported lazily, never
required.

Mono input, equal temperament. Use a hop divisible by `2 ** (n_octaves - 1)`
(e.g. 512 or 1024) — an odd hop disables the recursive decimation and inflates
memory (you'll get a warning).

## Performance

`fastchroma` is typically **3–15× faster** than other Python implementations
on the CPU, and **40–120× faster** with the GPU backends, with numerically
equivalent output (matched to within float32 rounding).

## Backends

| Platform | FFT (CPU) | GPU |
|----------|-----------|-----|
| macOS    | Accelerate / vDSP | Metal (Apple silicon) |
| Linux, Windows | PFFFT (SIMD) | CUDA (NVIDIA) |

On machines where a GPU backend is available (`fastchroma.gpu_available()`),
the default `backend="auto"` runs the CQT (all output variants, including
complex) and chroma on the GPU, and falls back to the CPU otherwise —
including for parameter combinations the GPU path does not support (e.g.
hops without enough factors of two). Force a path with `backend="cpu"` /
`backend="gpu"` (or the explicit vendor names `"metal"` / `"cuda"`), or set
the `FASTCHROMA_BACKEND` environment variable. The backends agree to ~1e-6
relative; bit-exactness across backends (or across platforms) is not
guaranteed.

Override the FFT at build time with `-DFASTCHROMA_FFT=pffft`, and the GPU
backend with `-DFASTCHROMA_GPU=metal|cuda|none` (default: Metal on macOS,
CUDA where `nvcc` is found, none otherwise). Linux and Windows wheels ship
with the CUDA backend built in (SASS for sm_60–sm_120 plus PTX, statically
linked cudart): at runtime they need only the NVIDIA driver — no CUDA
toolkit — and fall back to the CPU path on machines without one. Building
from source needs a CUDA toolkit for the CUDA backend; without one the
build is CPU-only.

## See also

fastchroma is part of the auvux DSP suite, re-exported through the `auvux.dsp`
façade:

- [fastmel](https://github.com/auvux/fastmel) — fused mel spectrogram, MFCC,
  and friends (`melspec`, `mfcc`).
- [metricon](https://github.com/auvux/metricon) — audio analysis metrics.

## License

MIT. Bundles PFFFT (BSD-style, FFTPACK); see `THIRD_PARTY_LICENSES`.
