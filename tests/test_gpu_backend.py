"""GPU backend (Metal or CUDA) vs the CPU path, through the public API.

Skipped wherever no GPU backend can run (CPU-only build, no usable device).
"""
import threading
import warnings

import numpy as np
import pytest

import fastchroma

pytestmark = pytest.mark.skipif(not fastchroma.gpu_available(),
                                reason="GPU backend not available")

SR = 22050
# Both backends use the exact same filterbank values; differences come only
# from FFT/decimation float rounding, observed ~5e-7.
TOL = 2e-5


def signal(seconds=3.0, odd=False):
    rng = np.random.default_rng(7)
    t = np.arange(int(SR * seconds)) / SR
    y = np.zeros_like(t)
    for i, f in enumerate((110.0, 220.0, 392.0, 587.33), start=1):
        y += np.sin(2 * np.pi * f * t) / i
    y += 0.01 * rng.standard_normal(t.shape)
    y = (y / np.max(np.abs(y))).astype(np.float32)
    if odd and len(y) % 2 == 0:
        y = y[:-1]
    return y


def assert_close(gpu, cpu):
    assert gpu.shape == cpu.shape
    peak = float(np.max(np.abs(cpu)))
    assert float(np.max(np.abs(gpu - cpu))) / peak < TOL


@pytest.mark.parametrize("bpo,n_bins", [
    (12, 84),    # defaults, 7 full octaves
    (12, 80),    # partial bottom octave
    (12, 24),    # few octaves -> exercises early downsampling
    (24, 168),
    (36, 108),
    (12, 8),     # single partial octave
])
def test_cqt_matches_cpu(bpo, n_bins):
    y = signal()
    kw = dict(sr=SR, hop_length=512, n_bins=n_bins, bins_per_octave=bpo)
    assert_close(fastchroma.cqt(y, **kw, backend="gpu"),
                 fastchroma.cqt(y, **kw, backend="cpu"))


@pytest.mark.parametrize("fmin", [55.0, 65.40639132514966])  # A1, C2
def test_cqt_fmin(fmin):
    y = signal()
    kw = dict(sr=SR, hop_length=512, n_bins=72, bins_per_octave=12, fmin=fmin)
    assert_close(fastchroma.cqt(y, **kw, backend="gpu"),
                 fastchroma.cqt(y, **kw, backend="cpu"))


@pytest.mark.parametrize("hop", [256, 1024])
@pytest.mark.parametrize("odd", [False, True])
def test_cqt_hop_and_odd_length(hop, odd):
    y = signal(2.37, odd=odd)
    kw = dict(sr=SR, hop_length=hop, n_bins=72, bins_per_octave=12)
    assert_close(fastchroma.cqt(y, **kw, backend="gpu"),
                 fastchroma.cqt(y, **kw, backend="cpu"))


def test_power_and_db_output():
    y = signal(1.0)
    m = fastchroma.cqt(y, sr=SR, backend="gpu")
    p = fastchroma.cqt(y, sr=SR, backend="gpu", output="power")
    db = fastchroma.cqt(y, sr=SR, backend="gpu", output="db")
    # power is computed in-kernel as ar^2+ai^2; magnitude is its sqrt, so the
    # round trip differs by float rounding only.
    assert np.allclose(p, m * m, rtol=1e-6, atol=1e-12)
    assert np.allclose(db, 10.0 * np.log10(np.maximum(p, 1e-10)), rtol=1e-5, atol=1e-4)


def test_complex_gpu_matches_cpu():
    y = signal()
    z_gpu = fastchroma.cqt(y, sr=SR, output="complex", backend="gpu")
    z_cpu = fastchroma.cqt(y, sr=SR, output="complex", backend="cpu")
    assert z_gpu.dtype == np.complex64 and z_gpu.shape == z_cpu.shape
    peak = float(np.max(np.abs(z_cpu)))
    assert float(np.max(np.abs(z_gpu - z_cpu))) / peak < TOL


@pytest.mark.parametrize("bpo,n_octaves", [(12, 7), (24, 7), (36, 7), (12, 5)])
def test_chroma_matches_cpu(bpo, n_octaves):
    y = signal()
    kw = dict(sr=SR, hop_length=512, bins_per_octave=bpo, n_octaves=n_octaves)
    gpu = fastchroma.chroma(y, **kw, backend="gpu")
    cpu = fastchroma.chroma(y, **kw, backend="cpu")
    # Chroma frames are L-infinity normalized, so compare absolutely.
    assert gpu.shape == cpu.shape
    assert float(np.max(np.abs(gpu - cpu))) < TOL


def test_auto_falls_back_on_unsupported_hop():
    # hop=100 has only two factors of 2; 7 octaves need six. auto must fall
    # back to the CPU path and still produce the CPU result.
    y = signal(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # odd-hop performance warning
        auto = fastchroma.cqt(y, sr=SR, hop_length=100, backend="auto")
        cpu = fastchroma.cqt(y, sr=SR, hop_length=100, backend="cpu")
    assert np.array_equal(auto, cpu)


def test_forced_gpu_raises_on_unsupported_hop():
    y = signal(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="gpu"):
            fastchroma.cqt(y, sr=SR, hop_length=100, backend="gpu")


def test_complex_auto_runs():
    z = fastchroma.cqt(signal(1.0), sr=SR, output="complex", backend="auto")
    assert z.dtype == np.complex64


def test_bad_backend_name():
    with pytest.raises(ValueError, match="backend"):
        fastchroma.cqt(signal(1.0), sr=SR, backend="opencl")


def test_vendor_backend_names():
    # The vendor name this build carries works like "gpu"; the other raises.
    mine = fastchroma.gpu_backend()
    other = {"metal": "cuda", "cuda": "metal"}[mine]
    y = signal(1.0)
    assert_close(fastchroma.cqt(y, sr=SR, backend=mine),
                 fastchroma.cqt(y, sr=SR, backend="cpu"))
    with pytest.raises(RuntimeError, match=other):
        fastchroma.cqt(y, sr=SR, backend=other)


def test_threaded_calls_match():
    """Concurrent calls — same engine, different engines, varying lengths."""
    ys = [signal(s) for s in (1.0, 2.0, 3.0)]
    jobs = [(i, y, bpo) for i, y in enumerate(ys) for bpo in (12, 24)] * 2
    expected = {
        (i, bpo): fastchroma.cqt(y, sr=SR, hop_length=512, n_bins=bpo * 7,
                                 bins_per_octave=bpo, backend="cpu")
        for i, y in enumerate(ys) for bpo in (12, 24)
    }

    results = [None] * len(jobs)
    errors = []

    def work(slot, y, bpo):
        try:
            results[slot] = fastchroma.cqt(y, sr=SR, hop_length=512, n_bins=bpo * 7,
                                           bins_per_octave=bpo, backend="gpu")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(s, y, bpo))
               for s, (i, y, bpo) in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for slot, (i, y, bpo) in enumerate(jobs):
        assert_close(results[slot], expected[(i, bpo)])
