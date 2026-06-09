// FFT backends: vDSP on Apple, PFFFT everywhere else.
//
// Both yield the unnormalized forward DFT. The two may differ by a conjugation
// of the imaginary part, which never matters here: fastchroma only emits
// magnitudes, and the filter basis and the signal always go through the same
// backend within a build.
#pragma once

#include <algorithm>
#include <complex>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace auvux::fastchroma {

using cf = std::complex<float>;

}  // namespace auvux::fastchroma

#if defined(__APPLE__) && !defined(FASTCHROMA_NO_ACCELERATE) && !defined(FASTCHROMA_FORCE_PFFFT)
#define FASTCHROMA_ACCELERATE 1
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include "pffft.h"
}
#endif

namespace auvux::fastchroma::fft {

constexpr int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

constexpr int ilog2(int n) {
    int k = 0;
    while ((1 << k) < n) ++k;
    return k;
}

#ifdef FASTCHROMA_ACCELERATE

inline FFTSetup vdsp_setup(int log2n) {
    static std::unordered_map<int, FFTSetup> cache;
    static std::mutex mutex;
    const std::lock_guard lock(mutex);
    auto [it, inserted] = cache.try_emplace(log2n, nullptr);
    if (inserted) it->second = vDSP_create_fftsetup(log2n, kFFTRadix2);
    return it->second;
}

// real[n] -> re[n/2+1], im[n/2+1]
class RealFFT {
public:
    explicit RealFFT(int n)
        : n_{n}, log2n_{ilog2(n)}, half_{n / 2 + 1}, setup_{vdsp_setup(log2n_)},
          re_(n / 2), im_(n / 2) {}

    void forward(const float* in, float* re, float* im) const {
        DSPSplitComplex split{re_.data(), im_.data()};
        vDSP_ctoz(reinterpret_cast<const DSPComplex*>(in), 2, &split, 1, n_ / 2);
        vDSP_fft_zrip(setup_, &split, 1, log2n_, kFFTDirection_Forward);
        // zrip returns twice the DFT; bin 0 and Nyquist are packed into [0].
        re[0] = re_[0] * 0.5f;
        im[0] = 0.0f;
        re[half_ - 1] = im_[0] * 0.5f;
        im[half_ - 1] = 0.0f;
        for (int k = 1; k < n_ / 2; ++k) {
            re[k] = re_[k] * 0.5f;
            im[k] = im_[k] * 0.5f;
        }
    }

private:
    int n_, log2n_, half_;
    FFTSetup setup_;
    mutable std::vector<float> re_, im_;
};

// complex[n] -> out[n/2+1]
class CplxFFT {
public:
    explicit CplxFFT(int n) : n_{n}, log2n_{ilog2(n)}, setup_{vdsp_setup(log2n_)}, re_(n), im_(n) {}

    void forward(const cf* in, cf* out) const {
        for (int i = 0; i < n_; ++i) {
            re_[i] = in[i].real();
            im_[i] = in[i].imag();
        }
        DSPSplitComplex split{re_.data(), im_.data()};
        vDSP_fft_zip(setup_, &split, 1, log2n_, kFFTDirection_Forward);
        for (int k = 0; k <= n_ / 2; ++k) out[k] = {re_[k], im_[k]};
    }

private:
    int n_, log2n_;
    FFTSetup setup_;
    mutable std::vector<float> re_, im_;
};

#else  // PFFFT

struct PffftDeleter {
    void operator()(float* p) const noexcept { pffft_aligned_free(p); }
};
using PffftBuffer = std::unique_ptr<float, PffftDeleter>;

inline PffftBuffer pffft_alloc(std::size_t floats) {
    return PffftBuffer{static_cast<float*>(pffft_aligned_malloc(floats * sizeof(float)))};
}

// Setups are read-only after creation, so they are shared across threads;
// the work buffers below are per-instance (one instance per worker thread).
inline PFFFT_Setup* pffft_setup(int n, pffft_transform_t type) {
    static std::unordered_map<long long, PFFFT_Setup*> cache;
    static std::mutex mutex;
    const long long key = (static_cast<long long>(n) << 1) | (type == PFFFT_REAL ? 0 : 1);
    const std::lock_guard lock(mutex);
    auto [it, inserted] = cache.try_emplace(key, nullptr);
    if (inserted) it->second = pffft_new_setup(n, type);
    return it->second;
}

class RealFFT {
public:
    explicit RealFFT(int n)
        : n_{n}, half_{n / 2 + 1}, setup_{pffft_setup(n, PFFFT_REAL)},
          in_{pffft_alloc(n)}, out_{pffft_alloc(n)}, work_{pffft_alloc(n)} {}

    void forward(const float* in, float* re, float* im) const {
        std::copy_n(in, n_, in_.get());  // PFFFT requires aligned input
        pffft_transform_ordered(setup_, in_.get(), out_.get(), work_.get(), PFFFT_FORWARD);
        // ordered real packing: [DC, Nyquist, re1, im1, re2, im2, ...]
        const float* o = out_.get();
        re[0] = o[0];
        im[0] = 0.0f;
        re[half_ - 1] = o[1];
        im[half_ - 1] = 0.0f;
        for (int k = 1; k < n_ / 2; ++k) {
            re[k] = o[2 * k];
            im[k] = o[2 * k + 1];
        }
    }

private:
    int n_, half_;
    PFFFT_Setup* setup_;
    PffftBuffer in_, out_, work_;
};

class CplxFFT {
public:
    explicit CplxFFT(int n)
        : n_{n}, setup_{pffft_setup(n, PFFFT_COMPLEX)},
          in_{pffft_alloc(2 * n)}, out_{pffft_alloc(2 * n)}, work_{pffft_alloc(2 * n)} {}

    void forward(const cf* in, cf* out) const {
        std::copy_n(reinterpret_cast<const float*>(in), 2 * n_, in_.get());
        pffft_transform_ordered(setup_, in_.get(), out_.get(), work_.get(), PFFFT_FORWARD);
        const float* o = out_.get();
        for (int k = 0; k <= n_ / 2; ++k) out[k] = {o[2 * k], o[2 * k + 1]};
    }

private:
    int n_;
    PFFFT_Setup* setup_;
    PffftBuffer in_, out_, work_;
};

#endif

}  // namespace auvux::fastchroma::fft
