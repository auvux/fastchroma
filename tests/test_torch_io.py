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
def test_torch_cuda_rejected():
    y = torch.from_numpy(signal_np()).cuda()
    with pytest.raises(TypeError, match="cuda"):
        fastchroma.cqt(y, sr=SR)


def test_gradient_tensors_accepted():
    y = torch.from_numpy(signal_np()).requires_grad_(True)
    out = fastchroma.cqt(y, sr=SR)  # detach() internally; no autograd through C++
    assert isinstance(out, torch.Tensor)
