# fastchroma

Fast constant-Q transform and chromagram for Python, with a small C++ core.

The constant-Q transform (Brown 1991; Schörkhuber & Klapuri 2010) and the
chromagram (Fujishima 1999) are standard music-analysis features. `fastchroma`
implements them directly in C++ with a per-platform FFT backend and a recursive
octave decimation that keeps memory bounded. It is CPU-only and its only
runtime dependency is NumPy (inputs and outputs are NumPy arrays).

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

One-shot functional forms exist too:

```python
mag    = fastchroma.cqt_magnitude(y, sr=22050, n_bins=84)
z      = fastchroma.cqt_complex(y, sr=22050)
chroma = fastchroma.chroma_cqt(y, sr=22050)
```

Mono input, equal temperament. Use a hop divisible by `2 ** (n_octaves - 1)`
(e.g. 512 or 1024) — an odd hop disables the recursive decimation and inflates
memory (you'll get a warning).

## Performance

`fastchroma` is typically **3–15× faster** than other Python implementations
and uses **less memory**, with numerically equivalent output (matched to within
float32 rounding).

## Backends

| Platform | FFT backend |
|----------|-------------|
| macOS    | Accelerate / vDSP |
| Linux, Windows | PFFFT (SIMD) |

Override at build time with `-DFASTCHROMA_FFT=pffft`.

## License

MIT. Bundles PFFFT (BSD-style, FFTPACK); see `THIRD_PARTY_LICENSES`.
