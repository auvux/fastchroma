#!/usr/bin/env python3
"""Benchmark fastchroma against other CQT / chroma libraries.

Speed and accuracy across audio lengths and bin densities, for the chromagram
and/or the raw CQT. Synthetic audio is generated internally (no files needed).
Whatever libraries are importable are included; accuracy is reported vs librosa.

    python benchmark.py                          # both modes, default lengths + bins
    python benchmark.py --mode cqt --bpo 12 36
    python benchmark.py --durations 30 120 --repeat 8
    python benchmark.py --all                    # also essentia + spafe (slow!)

Default impls: fastchroma, librosa, and (CQT only) nnAudio.
--all adds essentia and spafe — these are framewise / pure-Python and much
slower, so use short --durations with them.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

C1_HZ = 32.70319566257483


def make_signal(seconds, sr):
    rng = np.random.default_rng(0)
    t = np.arange(int(sr * seconds)) / sr
    y = np.zeros_like(t)
    for f0 in (65.41, 98.0, 130.81, 196.0, 261.63):  # a few notes + harmonics
        for h in (1, 2, 3):
            y += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    y += 0.01 * rng.standard_normal(t.shape)
    return (y / np.max(np.abs(y))).astype(np.float32)


def _essentia_accepts(cq, size):
    try:
        cq(np.zeros(size, dtype=np.float32))
        return True
    except Exception:  # noqa: BLE001
        return False


def build_impls(mode, sr, hop, bpo, n_octaves, n_bins, include_extra=False):
    """{name: callable(y) -> (rows, frames) ndarray} for the libs installed."""
    impls = {}

    try:
        import fastchroma
        if not hasattr(fastchroma, "chroma"):
            raise ImportError(
                f"installed fastchroma {getattr(fastchroma, '__version__', '?')} predates "
                "the cqt()/chroma() API — reinstall from this source tree: pip install .")

        def _fc(y, backend, _mode=mode):
            if _mode == "chroma":
                return fastchroma.chroma(y, sr=sr, hop_length=hop, bins_per_octave=bpo,
                                         n_octaves=n_octaves, backend=backend)
            return fastchroma.cqt(y, sr=sr, hop_length=hop, n_bins=n_bins,
                                  bins_per_octave=bpo, backend=backend)

        impls["fastchroma-cpu"] = lambda y: _fc(y, "cpu")
        if getattr(fastchroma, "metal_available", lambda: False)():
            impls["fastchroma-gpu"] = lambda y: _fc(y, "metal")
        elif sys.platform == "darwin":
            print("  note: metal backend unavailable on this machine", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"  note: fastchroma unavailable ({e})", file=sys.stderr)

    try:
        import librosa
        if mode == "chroma":
            impls["librosa"] = lambda y: librosa.feature.chroma_cqt(
                y=y, sr=sr, hop_length=hop, bins_per_octave=bpo, n_octaves=n_octaves, tuning=0.0)
        else:
            impls["librosa"] = lambda y: np.abs(librosa.cqt(
                y=y, sr=sr, hop_length=hop, n_bins=n_bins, bins_per_octave=bpo, tuning=0.0))
    except Exception:  # noqa: BLE001
        pass

    if mode != "cqt":
        return impls  # the libs below are CQT-only

    try:  # nnAudio (PyTorch)
        import torch
        try:
            from nnAudio.features.cqt import CQT
        except Exception:  # noqa: BLE001
            from nnAudio.Spectrogram import CQT
        net = CQT(sr=sr, hop_length=hop, n_bins=n_bins, bins_per_octave=bpo,
                  fmin=C1_HZ, verbose=False, output_format="Magnitude")
        impls["nnAudio"] = lambda y: net(torch.tensor(y).unsqueeze(0)).numpy()[0]
    except Exception:  # noqa: BLE001
        pass

    if include_extra:
        try:  # essentia: framewise ConstantQ (C++), comparable hop-based framing
            import essentia
            essentia.log.infoActive = False
            essentia.log.warningActive = False
            import essentia.standard as es
            cq = es.ConstantQ(minFrequency=C1_HZ, numberBins=n_bins,
                              binsPerOctave=bpo, sampleRate=sr)
            fsz = next((1 << k for k in range(10, 18) if _essentia_accepts(cq, 1 << k)), None)
            if fsz:
                def _ess(y, _cq=cq, _fsz=fsz):
                    import essentia.standard as es2
                    frames = es2.FrameGenerator(y, frameSize=_fsz, hopSize=hop, startFromZero=True)
                    return np.array([np.abs(_cq(f)) for f in frames]).T
                impls["essentia"] = _ess
        except Exception:  # noqa: BLE001
            pass

        try:  # spafe (pure Python)
            from spafe.features.cqcc import cqt_spectrogram
            def _spafe(y):
                out = cqt_spectrogram(y, fs=sr, low_freq=C1_HZ, number_of_octaves=n_octaves,
                                      number_of_bins_per_octave=bpo, pre_emph=False)
                a = np.asarray(out[0] if isinstance(out, tuple) else out)
                return a if a.shape[0] == n_bins else a.T
            impls["spafe"] = _spafe
        except Exception:  # noqa: BLE001
            pass

    return impls


def best_time(fn, repeat):
    fn()  # warm-up (builds filterbanks / kernels)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def accuracy(out, ref):
    out, ref = np.asarray(out), np.asarray(ref)
    if out.ndim != 2 or ref.ndim != 2 or out.shape[0] != ref.shape[0]:
        return None, None  # different binning/shape — not comparable
    n = min(out.shape[1], ref.shape[1])
    a = out[:, :n].astype(float)
    b = (np.abs(ref[:, :n]) if np.iscomplexobj(ref) else ref[:, :n]).astype(float)
    # max-normalize each so relerr reflects structure, not a constant scale factor
    a = a / (float(np.max(np.abs(a))) or 1.0)
    b = b / (float(np.max(np.abs(b))) or 1.0)
    return (float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), float(np.max(np.abs(a - b))))


def run_table(mode, bpo, n_octaves, durations, sr, hop, repeat, include_extra):
    n_bins = bpo * n_octaves
    impls = build_impls(mode, sr, hop, bpo, n_octaves, n_bins, include_extra)
    if not impls:
        return
    extra = f"n_bins={n_bins}" if mode == "cqt" else "n_chroma=12"
    print(f"[{mode}]  bpo={bpo}  hop={hop}  {extra}   impls: {', '.join(impls)}")
    cols = "{:>7} {:<14} {:>10} {:>8} {:>8} {:>9}"
    print(cols.format("dur", "impl", "time", "speedup", "corr", "relerr"))
    print("-" * 62)
    for dur in durations:
        y = make_signal(dur, sr)
        ref = impls["librosa"](y) if "librosa" in impls else None
        timings = {name: best_time(lambda f=fn: f(y), repeat) for name, fn in impls.items()}
        base = timings.get("librosa")
        for name, fn in impls.items():
            t = timings[name]
            corr, relerr = (accuracy(fn(y), ref) if (ref is not None and name != "librosa") else (None, None))
            speed = f"{base / t:.1f}x" if base else "-"
            cc = f"{corr:.4f}" if corr is not None else "-"
            re = f"{relerr:.1e}" if relerr is not None else "-"
            print(cols.format(f"{dur:g}s", name, f"{t*1000:.2f}ms", speed, cc, re))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["chroma", "cqt", "both"], default="both")
    ap.add_argument("--durations", type=float, nargs="+", default=[10, 30, 60, 120, 300])
    ap.add_argument("--bpo", type=int, nargs="+", default=[12, 24, 36],
                    help="bins per octave to sweep")
    ap.add_argument("--n-octaves", type=int, default=7)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--all", action="store_true", help="also benchmark essentia + spafe (slow)")
    args = ap.parse_args()

    print(f"python {sys.version.split()[0]}  numpy {np.__version__}  repeat={args.repeat}")
    print("accuracy vs librosa (max-normalized): corr + relerr; '-' = different binning\n")

    modes = ["chroma", "cqt"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for bpo in args.bpo:
            run_table(mode, bpo, args.n_octaves, args.durations, args.sr, args.hop,
                      args.repeat, args.all)


if __name__ == "__main__":
    main()
