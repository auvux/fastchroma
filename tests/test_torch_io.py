"""Input/output type mirroring: numpy in -> numpy out, torch in -> torch out
(same device), like fastmel. Skipped when torch is not installed."""
import numpy as np
import pytest

import fastchroma

torch = pytest.importorskip("torch")

SR = 22050


def signal_np(seconds=2.0):
    rng = np.random.default_rng(3)
    t = np.arange(int(SR * seconds)) / SR
    y = np.sin(2 * np.pi * 220.0 * t) + 0.01 * rng.standard_normal(t.shape)
    return y.astype(np.float32)


def test_numpy_in_numpy_out():
    out = fastchroma.cqt(signal_np(), sr=SR)
    assert isinstance(out, np.ndarray)


def test_torch_cpu_mirror():
    y = signal_np()
    out_t = fastchroma.cqt(torch.from_numpy(y), sr=SR)
    assert isinstance(out_t, torch.Tensor) and out_t.device.type == "cpu"
    assert out_t.dtype == torch.float32
    assert np.array_equal(out_t.numpy(), fastchroma.cqt(y, sr=SR))

    ch = fastchroma.chroma(torch.from_numpy(y), sr=SR)
    assert isinstance(ch, torch.Tensor)
    assert np.array_equal(ch.numpy(), fastchroma.chroma(y, sr=SR))


def test_torch_complex_mirror():
    y = signal_np()
    z = fastchroma.cqt(torch.from_numpy(y), sr=SR, output="complex")
    assert isinstance(z, torch.Tensor) and z.dtype == torch.complex64


def test_torch_float64_converts():
    y = torch.from_numpy(signal_np().astype(np.float64))
    out = fastchroma.cqt(y, sr=SR)
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_torch_mps_mirror():
    y = signal_np()
    y_mps = torch.from_numpy(y).to("mps")
    out = fastchroma.cqt(y_mps, sr=SR)
    assert isinstance(out, torch.Tensor) and out.device.type == "mps"
    ref = fastchroma.cqt(y, sr=SR)
    assert np.allclose(out.cpu().numpy(), ref, rtol=1e-5, atol=1e-6)

    ch = fastchroma.chroma(y_mps, sr=SR)
    assert ch.device.type == "mps"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_torch_cuda_mirror():
    # On CUDA builds this is the device-resident (DLPack) path: no host
    # round trip, result allocated on the GPU.
    y = signal_np()
    y_cuda = torch.from_numpy(y).cuda()
    out = fastchroma.cqt(y_cuda, sr=SR)
    assert isinstance(out, torch.Tensor) and out.device.type == "cuda"
    ref = fastchroma.cqt(y, sr=SR)
    assert np.allclose(out.cpu().numpy(), ref, rtol=1e-5, atol=1e-6)

    ch = fastchroma.chroma(y_cuda, sr=SR)
    assert ch.device.type == "cuda"
    ch_ref = fastchroma.chroma(y, sr=SR)
    assert np.allclose(ch.cpu().numpy(), ch_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_torch_cuda_complex():
    y = signal_np()
    z = fastchroma.cqt(torch.from_numpy(y).cuda(), sr=SR, output="complex")
    assert isinstance(z, torch.Tensor) and z.device.type == "cuda"
    assert z.dtype == torch.complex64
    ref = fastchroma.cqt(y, sr=SR, output="complex", backend="cpu")
    peak = float(np.max(np.abs(ref)))
    assert float(np.max(np.abs(z.cpu().numpy() - ref))) / peak < 2e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_torch_cuda_unsupported_hop_falls_back():
    # hop=100 lacks the factors of two the GPU path needs; under auto the
    # resident path must fall back (through host staging, to the CPU) and
    # still return a CUDA tensor.
    import warnings
    y = signal_np()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # odd-hop performance warning
        out = fastchroma.cqt(torch.from_numpy(y).cuda(), sr=SR, hop_length=100)
        ref = fastchroma.cqt(y, sr=SR, hop_length=100, backend="cpu")
    assert out.device.type == "cuda"
    assert np.array_equal(out.cpu().numpy(), ref)


def test_gradient_tensors_accepted():
    y = torch.from_numpy(signal_np()).requires_grad_(True)
    out = fastchroma.cqt(y, sr=SR)  # detach() internally; no autograd through C++
    assert isinstance(out, torch.Tensor)
