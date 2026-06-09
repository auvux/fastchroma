import numpy as np
import pytest

import fastchroma

SR = 22050


def signal(seconds=8.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * seconds)) / SR
    y = sum((1.0 / h) * np.sin(2 * np.pi * f * h * t)
            for f in (65.41, 98.0, 130.81, 196.0, 261.63) for h in (1, 2, 3))
    y += 0.01 * rng.standard_normal(t.shape)
    return (y / np.max(np.abs(y))).astype(np.float32)


def test_shapes():
    y = signal(4.0)
    chroma = fastchroma.chroma_cqt(y, sr=SR, hop_length=512)
    assert chroma.shape[0] == 12
    cqt = fastchroma.cqt_magnitude(y, sr=SR, hop_length=512, n_bins=84)
    assert cqt.shape[0] == 84
    assert chroma.shape[1] == cqt.shape[1]
    assert np.all(np.isfinite(chroma)) and np.all(chroma >= 0)


def test_object_api():
    y = signal(4.0)
    cqt = fastchroma.CQT(sr=SR, hop_length=512, n_bins=84, bins_per_octave=12)
    assert cqt.frequencies.shape == (84,)
    assert np.allclose(cqt.frequencies[0], 32.70319566, rtol=1e-6)
    assert np.array_equal(cqt(y), fastchroma.cqt_magnitude(y, sr=SR, hop_length=512))
    assert np.array_equal(cqt(y, output="power"), cqt(y) ** 2)
    chroma = fastchroma.Chroma(sr=SR, hop_length=512)(y)
    assert np.array_equal(chroma, fastchroma.chroma_cqt(y, sr=SR, hop_length=512))


def test_complex_output():
    y = signal(4.0)
    cqt = fastchroma.CQT(sr=SR, hop_length=512, n_bins=84, bins_per_octave=12)
    z = cqt(y, output="complex")
    assert z.dtype == np.complex64
    # magnitude of the complex output equals the magnitude transform
    assert np.allclose(np.abs(z), cqt(y), rtol=1e-5, atol=1e-6)


def test_complex_matches_librosa():
    librosa = pytest.importorskip("librosa")
    y = signal()
    z = fastchroma.cqt_complex(y, sr=SR, hop_length=512, n_bins=84, bins_per_octave=12)
    ref = librosa.cqt(y=y, sr=SR, hop_length=512, n_bins=84, bins_per_octave=12,
                      tuning=0.0).astype(np.complex64)
    n = min(z.shape[1], ref.shape[1])
    rel = np.max(np.abs(z[:, :n] - ref[:, :n])) / np.max(np.abs(ref))
    assert rel < 5e-3


def test_odd_hop_warns():
    with pytest.warns(UserWarning):
        fastchroma.CQT(sr=SR, hop_length=441, n_bins=84, bins_per_octave=12)


@pytest.mark.parametrize("bins_per_octave", [12, 24, 36])
def test_matches_librosa(bins_per_octave):
    librosa = pytest.importorskip("librosa")
    y = signal()
    got = fastchroma.chroma_cqt(y, sr=SR, hop_length=512,
                                bins_per_octave=bins_per_octave, n_octaves=7)
    ref = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512,
                                     bins_per_octave=bins_per_octave, n_octaves=7,
                                     tuning=0.0).astype(np.float32)
    n = min(got.shape[1], ref.shape[1])
    rel = np.max(np.abs(got[:, :n] - ref[:, :n])) / np.max(np.abs(ref))
    corr = np.corrcoef(got[:, :n].ravel(), ref[:, :n].ravel())[0, 1]
    assert corr > 0.999
    assert rel < 5e-3
