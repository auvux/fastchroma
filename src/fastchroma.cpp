// Constant-Q transform via recursive octave decimation
// (Schoerkhuber & Klapuri, 2010), folded to pitch classes for the chromagram
// (Fujishima, 1999). Memory stays bounded because each octave halves both the
// signal and the hop, so every per-octave FFT keeps the same small size.
#include "fastchroma.hpp"

#include "cqt_internal.hpp"
#include "fft_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <thread>
#include <tuple>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace auvux::fastchroma {
namespace {

// Filterbank construction, half-band design, and the chroma fold live in
// cqt_internal.hpp, shared with the Metal backend.
using namespace detail;
using fft::RealFFT;

// Allocator that aligns to a cache line and skips value-initialization for
// buffers we overwrite entirely.
template <class T, std::size_t Align = 64>
struct AlignedNoInit {
    using value_type = T;
    template <class U> struct rebind { using other = AlignedNoInit<U, Align>; };

    AlignedNoInit() = default;
    template <class U> AlignedNoInit(const AlignedNoInit<U, Align>&) noexcept {}

    T* allocate(std::size_t n) {
        const std::size_t bytes = (n ? n : 1) * sizeof(T);
#if defined(_MSC_VER)
        void* p = _aligned_malloc(bytes, Align);
        if (!p) throw std::bad_alloc{};
#else
        void* p = nullptr;
        if (posix_memalign(&p, Align, bytes) != 0) throw std::bad_alloc{};
#endif
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
    template <class U> void construct(U*) {}  // leave uninitialized
    template <class U, class... Args> void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};
template <class T, class U, std::size_t A>
bool operator==(const AlignedNoInit<T, A>&, const AlignedNoInit<U, A>&) { return true; }
template <class T, class U, std::size_t A>
bool operator!=(const AlignedNoInit<T, A>&, const AlignedNoInit<U, A>&) { return false; }

using FloatVec = std::vector<float, AlignedNoInit<float>>;

// Memoize per octave: identical (sr, freqs, scale, sparsity) recur across calls
// with the same config, so a batch of files builds each basis once.
const SparseBasis& cached_basis(const std::vector<double>& freqs, const std::vector<double>& alpha,
                                double sr, double base_sr, double filter_scale, double sparsity) {
    using Key = std::tuple<double, double, int, double, double, double, double>;
    static std::map<Key, SparseBasis> cache;
    static std::mutex mutex;
    const Key key{sr, base_sr, static_cast<int>(freqs.size()),
                  freqs.front(), freqs.back(), filter_scale, sparsity};
    const std::lock_guard lock(mutex);
    auto it = cache.find(key);
    if (it == cache.end())
        it = cache.emplace(key, build_basis(freqs, alpha, sr, base_sr, filter_scale, sparsity)).first;
    return it->second;
}

// Complex response of one octave, bin-major: out[r * n_frames + t].
struct Response {
    FloatVec re, im;
    int rows = 0, cols = 0;
};

// Fused per-frame STFT (boxcar window, centered) and sparse projection: each
// frame is transformed and immediately projected, so there is no full STFT
// matrix and no transpose. Frames are independent and run across threads.
Response octave_response(const float* signal, int n, int hop, const SparseBasis& basis) {
    const int n_fft = basis.n_fft;
    const int pad = n_fft / 2;
    const int half = n_fft / 2 + 1;
    const int n_frames = 1 + n / hop;

    Response out;
    out.rows = basis.rows;
    out.cols = n_frames;
    out.re.resize(static_cast<std::size_t>(basis.rows) * n_frames);
    out.im.resize(static_cast<std::size_t>(basis.rows) * n_frames);

    const int* const row_start = basis.row_start.data();
    const int* const col = basis.col.data();
    const cf* const val = basis.val.data();

    const auto process = [&](int from, int to) {
        RealFFT rfft(n_fft);
        std::vector<float> re(half), im(half), frame(n_fft);
        for (int t = from; t < to; ++t) {
            const int start = t * hop - pad;
            const float* in;
            if (start >= 0 && start + n_fft <= n) {
                in = signal + start;  // fully inside: read in place
            } else {
                for (int k = 0; k < n_fft; ++k) {
                    const int idx = start + k;
                    frame[k] = (idx >= 0 && idx < n) ? signal[idx] : 0.0f;
                }
                in = frame.data();
            }
            rfft.forward(in, re.data(), im.data());
            for (int r = 0; r < basis.rows; ++r) {
                float ar = 0.0f, ai = 0.0f;
                for (int idx = row_start[r]; idx < row_start[r + 1]; ++idx) {
                    const float vr = val[idx].real(), vi = val[idx].imag();
                    const float xr = re[col[idx]], xi = im[col[idx]];
                    ar += vr * xr - vi * xi;
                    ai += vr * xi + vi * xr;
                }
                out.re[static_cast<std::size_t>(r) * n_frames + t] = ar;
                out.im[static_cast<std::size_t>(r) * n_frames + t] = ai;
            }
        }
    };

    int n_threads = 1;
    if (n_frames >= 1024) {
        const char* env = std::getenv("FASTCHROMA_THREADS");
        n_threads = env ? std::atoi(env) : static_cast<int>(std::thread::hardware_concurrency());
        n_threads = std::clamp(n_threads, 1, std::min(24, n_frames / 512));
    }
    if (n_threads <= 1) {
        process(0, n_frames);
    } else {
        std::vector<std::thread> pool;
        const int chunk = (n_frames + n_threads - 1) / n_threads;
        for (int i = 0; i < n_threads; ++i) {
            const int from = i * chunk, to = std::min(n_frames, from + chunk);
            if (from < to) pool.emplace_back(process, from, to);
        }
        for (auto& t : pool) t.join();
    }
    return out;
}

// Decimate by two (then scale by sqrt(2), to match the analytic CQT scaling),
// writing into caller-owned buffers so the octave loop allocates nothing.
void decimate(const float* x, int n, std::vector<float>& out,
              FloatVec& even, FloatVec& odd, FloatVec& padded) {
    static const HalfBand hb;
    const int out_n = (n + 1) / 2;
    const int pairs = n / 2;

    even.resize(out_n);
    odd.resize(std::max(pairs, 1));
    for (int p = 0; p < pairs; ++p) {
        even[p] = x[2 * p];
        odd[p] = x[2 * p + 1];
    }
    if (n & 1) even[pairs] = x[n - 1];

    padded.resize(static_cast<std::size_t>(out_n) + 31);
    std::fill_n(padded.begin(), 16, 0.0f);
    const int copy = std::min(pairs, static_cast<int>(padded.size()) - 16);
    std::copy_n(odd.begin(), copy, padded.begin() + 16);
    std::fill(padded.begin() + 16 + copy, padded.end(), 0.0f);

    out.resize(out_n);
    constexpr float s2 = 1.4142135623730951f;
    for (int m = 0; m < out_n; ++m) {
        float acc = 0.0f;
        for (int t = 0; t < 32; ++t) acc += hb.odd[t] * padded[m + t];
        out[m] = (hb.center * even[m] + acc) * s2;
    }
}

}  // namespace

namespace {

// Complex CQT, scaled. cqt and cqt_complex both derive from this.
ComplexMatrix cqt_core(const float* y, std::size_t n, const CqtParams& p) {
    const double fmin = (p.fmin > 0.0) ? p.fmin : kC1Hz;
    const int n_octaves = static_cast<int>(std::ceil(static_cast<double>(p.n_bins) / p.bins_per_octave));
    const int n_filters = std::min(p.bins_per_octave, p.n_bins);

    const auto freqs = cqt_frequencies(p.n_bins, fmin, p.bins_per_octave);
    const auto alpha = relative_bandwidth(freqs);
    double cutoff = 0.0;
    wavelet_lengths(freqs, p.sr, alpha, p.filter_scale, &cutoff);

    double base_sr = p.sr;
    int hop = p.hop;

    // Octave 0 reads the caller's buffer; lower octaves ping-pong between two
    // owned buffers with shared decimation scratch — no per-octave allocation.
    const float* signal = y;
    int signal_len = static_cast<int>(n);
    std::array<std::vector<float>, 2> buffers;
    int write = 0;
    FloatVec scratch_even, scratch_odd, scratch_pad;
    const auto decimate_step = [&] {
        decimate(signal, signal_len, buffers[write], scratch_even, scratch_odd, scratch_pad);
        signal = buffers[write].data();
        signal_len = static_cast<int>(buffers[write].size());
        write ^= 1;
    };

    // Optional upfront downsampling when the top octave leaves headroom.
    const int down1 = std::max(0, static_cast<int>(std::ceil(std::log2(p.sr / 2.0 / cutoff))) - 2);
    const int down2 = std::max(0, two_factor_count(hop) - n_octaves + 1);
    for (int i = std::min(down1, down2); i > 0; --i) {
        decimate_step();
        base_sr /= 2.0;
        hop /= 2;
    }

    // All octaves share a frame count (signal and hop halve together); take the
    // min so each octave's output trims to the same width.
    int n_frames = std::numeric_limits<int>::max();
    for (int oc = 0, len = signal_len, h = hop; oc < n_octaves; ++oc) {
        n_frames = std::min(n_frames, 1 + len / h);
        if (h % 2 == 0) { h /= 2; len = (len + 1) / 2; }
    }

    ComplexMatrix out;
    out.rows = p.n_bins;
    out.cols = n_frames;
    out.re.assign(static_cast<std::size_t>(p.n_bins) * n_frames, 0.0f);
    out.im.assign(static_cast<std::size_t>(p.n_bins) * n_frames, 0.0f);

    double sr = base_sr;
    int octave_hop = hop;
    int row_end = p.n_bins;  // octave 0 occupies the top rows

    for (int oc = 0; oc < n_octaves; ++oc) {
        const int lo = std::max(0, p.n_bins - n_filters * (oc + 1));
        const int hi = (oc == 0) ? p.n_bins : p.n_bins - n_filters * oc;
        const std::vector<double> oct_freqs(freqs.begin() + lo, freqs.begin() + hi);
        const std::vector<double> oct_alpha(alpha.begin() + lo, alpha.begin() + hi);

        const SparseBasis& basis = cached_basis(oct_freqs, oct_alpha, sr, base_sr,
                                                p.filter_scale, p.sparsity);
        const Response r = octave_response(signal, signal_len, octave_hop, basis);

        // The bottom octave can be clamped to fewer than n_filters rows; align
        // the copy to the rows that actually exist.
        const int start_row = row_end - r.rows;
        for (int row = 0; row < r.rows; ++row) {
            const int dst = start_row + row;
            if (dst < 0 || dst >= p.n_bins) continue;
            const std::size_t src = static_cast<std::size_t>(row) * r.cols;
            const std::size_t out_off = static_cast<std::size_t>(dst) * n_frames;
            std::copy_n(&r.re[src], n_frames, &out.re[out_off]);
            std::copy_n(&r.im[src], n_frames, &out.im[out_off]);
        }
        row_end -= n_filters;

        if (octave_hop % 2 == 0) {
            decimate_step();
            octave_hop /= 2;
            sr /= 2.0;
        }
    }

    const auto lengths = wavelet_lengths(freqs, base_sr, alpha, p.filter_scale);
    for (int row = 0; row < p.n_bins; ++row) {
        const float s = static_cast<float>(1.0 / std::sqrt(lengths[row]));
        float* re = &out.re[static_cast<std::size_t>(row) * n_frames];
        float* im = &out.im[static_cast<std::size_t>(row) * n_frames];
        for (int t = 0; t < n_frames; ++t) { re[t] *= s; im[t] *= s; }
    }
    return out;
}

}  // namespace

Matrix cqt(const float* y, std::size_t n, const CqtParams& p) {
    const ComplexMatrix c = cqt_core(y, n, p);
    Matrix out;
    out.rows = c.rows;
    out.cols = c.cols;
    out.data.resize(c.re.size());
    for (std::size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = std::sqrt(c.re[i] * c.re[i] + c.im[i] * c.im[i]);
    return out;
}

ComplexMatrix cqt_complex(const float* y, std::size_t n, const CqtParams& p) {
    return cqt_core(y, n, p);
}

Matrix chroma(const float* y, std::size_t n, double sr, int hop, int bins_per_octave,
                  int n_octaves, int n_chroma, double fmin) {
    const double f0 = (fmin > 0.0) ? fmin : kC1Hz;
    CqtParams params;
    params.sr = sr;
    params.hop = hop;
    params.fmin = f0;
    params.n_bins = n_octaves * bins_per_octave;
    params.bins_per_octave = bins_per_octave;
    const Matrix mag = cqt(y, n, params);
    return detail::chroma_fold(mag, bins_per_octave, n_chroma, f0);
}

#ifndef FASTCHROMA_METAL
// Stubs for non-Apple builds; the real implementations are in metal_backend.mm.
bool metal_available() { return false; }
bool cqt_metal(const float*, std::size_t, const CqtParams&, Matrix&, int) { return false; }
bool cqt_complex_metal(const float*, std::size_t, const CqtParams&, ComplexMatrix&) {
    return false;
}
bool chroma_metal(const float*, std::size_t, double, int, int, int, int, double, Matrix&) {
    return false;
}
#endif

}  // namespace auvux::fastchroma
