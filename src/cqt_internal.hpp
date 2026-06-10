// Internal shared machinery: CQT filterbank construction, the half-band
// decimation filter design, and the chroma fold. Used by the CPU core
// (fastchroma.cpp) and the Metal backend (metal_backend.mm) — keeping both
// paths numerically identical by construction.
#pragma once

#include "fastchroma.hpp"
#include "fft_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace auvux::fastchroma::detail {

constexpr double kPi = 3.14159265358979323846;
// Hann equivalent noise bandwidth, ENBW = N*sum(w^2)/sum(w)^2 (Harris 1978) for a
// symmetric Hann window of length 8193; analytic continuous limit is 3/2.
constexpr double kHannEnbw = 1.50018310546875;
constexpr double kC1Hz = 32.70319566257483;  // note C1 = 440 * 2^((24-69)/12), A440 / 12-TET

inline int two_factor_count(int x) {
    int count = 0;
    for (; x > 0 && x % 2 == 0; x /= 2) ++count;
    return count;
}

inline std::vector<double> cqt_frequencies(int n_bins, double fmin, int bins_per_octave) {
    std::vector<double> freqs(n_bins);
    for (int k = 0; k < n_bins; ++k)
        freqs[k] = fmin * std::pow(2.0, static_cast<double>(k) / bins_per_octave);
    return freqs;
}

// Local relative bandwidth at each frequency, from the geometric spacing.
inline std::vector<double> relative_bandwidth(const std::vector<double>& freqs) {
    const int n = static_cast<int>(freqs.size());
    std::vector<double> logf(n), bpo(n), alpha(n);
    for (int i = 0; i < n; ++i) logf[i] = std::log2(freqs[i]);
    bpo.front() = 1.0 / (logf[1] - logf[0]);
    bpo.back() = 1.0 / (logf[n - 1] - logf[n - 2]);
    for (int i = 1; i < n - 1; ++i) bpo[i] = 2.0 / (logf[i + 1] - logf[i - 1]);
    for (int i = 0; i < n; ++i) {
        const double t = std::pow(2.0, 2.0 / bpo[i]);
        alpha[i] = (t - 1.0) / (t + 1.0);
    }
    return alpha;
}

inline std::vector<double> wavelet_lengths(const std::vector<double>& freqs, double sr,
                                           const std::vector<double>& alpha, double filter_scale,
                                           double* cutoff_out = nullptr) {
    std::vector<double> lengths(freqs.size());
    double cutoff = 0.0;
    for (std::size_t i = 0; i < freqs.size(); ++i) {
        const double Q = filter_scale / alpha[i];
        lengths[i] = Q * sr / freqs[i];
        cutoff = std::max(cutoff, freqs[i] * (1.0 + 0.5 * kHannEnbw / Q));
    }
    if (cutoff_out) *cutoff_out = cutoff;
    return lengths;
}

// One octave of the filterbank, FFT'd, sparsified, and pre-scaled. Stored CSR.
struct SparseBasis {
    int rows = 0;
    int n_fft = 0;
    std::vector<int> row_start;  // size rows + 1
    std::vector<int> col;
    std::vector<cf> val;
};

inline SparseBasis build_basis(const std::vector<double>& freqs, const std::vector<double>& alpha,
                               double sr, double base_sr, double filter_scale, double sparsity) {
    const int rows = static_cast<int>(freqs.size());
    const auto lengths = wavelet_lengths(freqs, sr, alpha, filter_scale);
    const int n_fft = fft::next_pow2(static_cast<int>(std::ceil(*std::max_element(lengths.begin(), lengths.end()))));
    const int half = n_fft / 2 + 1;
    const float scale = static_cast<float>(std::sqrt(base_sr / sr));

    SparseBasis basis;
    basis.rows = rows;
    basis.n_fft = n_fft;
    basis.row_start.assign(rows + 1, 0);

    std::vector<cf> time(n_fft), spec(half);
    fft::CplxFFT cfft(n_fft);
    std::vector<std::vector<std::pair<int, cf>>> sparse_rows(rows);

    for (int k = 0; k < rows; ++k) {
        const double len = lengths[k];
        const long lo = static_cast<long>(std::floor(-len / 2.0));
        const long hi = static_cast<long>(std::floor(len / 2.0));
        const int count = static_cast<int>(hi - lo);
        const int pad = (n_fft - count) / 2;

        // Windowed complex sinusoid, L1-normalized, then scaled for the FFT length.
        std::fill(time.begin(), time.end(), cf{});
        double l1 = 0.0;
        std::vector<cf> sig(count);
        for (int j = 0; j < count; ++j) {
            const double phase = (lo + j) * 2.0 * kPi * freqs[k] / sr;
            const double window = 0.5 - 0.5 * std::cos(2.0 * kPi * j / count);
            sig[j] = {static_cast<float>(std::cos(phase) * window),
                      static_cast<float>(std::sin(phase) * window)};
            l1 += std::abs(sig[j]);
        }
        const float norm = (l1 > 0.0 ? static_cast<float>(len / (l1 * n_fft)) : 1.0f);
        for (int j = 0; j < count; ++j) time[pad + j] = sig[j] * norm;

        cfft.forward(time.data(), spec.data());

        // Keep the bins carrying all but `sparsity` of the row's magnitude.
        std::vector<float> mags(half);
        double total = 0.0;
        for (int i = 0; i < half; ++i) total += (mags[i] = std::abs(spec[i]));
        std::vector<float> sorted = mags;
        std::sort(sorted.begin(), sorted.end());
        float threshold = sorted.empty() ? 0.0f : sorted.back();
        if (total > 0.0) {
            double cumulative = 0.0;
            for (int i = 0; i < half; ++i) {
                cumulative += sorted[i] / total;
                if (cumulative >= sparsity) { threshold = sorted[i]; break; }
            }
        }
        for (int i = 0; i < half; ++i)
            if (mags[i] >= threshold) sparse_rows[k].emplace_back(i, spec[i] * scale);
    }

    for (int k = 0; k < rows; ++k)
        basis.row_start[k + 1] = basis.row_start[k] + static_cast<int>(sparse_rows[k].size());
    basis.col.resize(basis.row_start[rows]);
    basis.val.resize(basis.row_start[rows]);
    for (int k = 0; k < rows; ++k) {
        int off = basis.row_start[k];
        for (const auto& [c, v] : sparse_rows[k]) {
            basis.col[off] = c;
            basis.val[off] = v;
            ++off;
        }
    }
    return basis;
}

// Linear-phase Kaiser half-band low-pass (fc = 0.25). In polyphase form the
// even-offset taps are exactly zero, so only the centre tap and the 32 odd
// taps remain.
struct HalfBand {
    static constexpr int kTaps = 65;
    float center;
    std::array<float, 32> odd;

    HalfBand() {
        constexpr double fc = 0.25, beta = 8.0;
        constexpr double mid = (kTaps - 1) / 2.0;
        const auto bessel_i0 = [](double x) {
            double sum = 1.0, term = 1.0;
            for (int k = 1; k < 40; ++k) { term *= (x * x) / (4.0 * k * k); sum += term; }
            return sum;
        };
        const double i0_beta = bessel_i0(beta);
        std::array<double, kTaps> h{};
        double dc = 0.0;
        for (int n = 0; n < kTaps; ++n) {
            const double m = n - mid;
            const double sinc = (m == 0.0) ? 2.0 * fc : std::sin(2.0 * kPi * fc * m) / (kPi * m);
            const double r = 2.0 * n / (kTaps - 1) - 1.0;
            const double window = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0_beta;
            h[n] = sinc * window;
            dc += h[n];
        }
        for (double& v : h) v /= dc;
        center = static_cast<float>(h[(kTaps - 1) / 2]);
        for (int t = 0; t < 32; ++t) odd[t] = static_cast<float>(h[2 * t + 1]);
    }
};

// Fold CQT bins onto pitch classes (tiled identity rolled so the lowest bin's
// pitch class lands on chroma 0), then L-infinity normalize each frame.
inline Matrix chroma_fold(const Matrix& cqt, int bins_per_octave, int n_chroma, double f0) {
    const int n_input = cqt.rows;
    const int merge = bins_per_octave / n_chroma;
    const int roll_cols = merge / 2;
    const double midi0 = std::fmod(12.0 * (std::log2(f0) - std::log2(440.0)) + 69.0, 12.0);
    int roll = static_cast<int>(std::lround(midi0 * (n_chroma / 12.0)));
    roll = ((roll % n_chroma) + n_chroma) % n_chroma;

    const auto weight = [&](int chroma, int bin) -> float {
        const int folded = (bin % bins_per_octave + roll_cols) % bins_per_octave;
        const int src = ((chroma - roll) % n_chroma + n_chroma) % n_chroma;
        return folded / merge == src ? 1.0f : 0.0f;
    };

    Matrix out;
    out.rows = n_chroma;
    out.cols = cqt.cols;
    out.data.assign(static_cast<std::size_t>(n_chroma) * cqt.cols, 0.0f);
    for (int c = 0; c < n_chroma; ++c) {
        float* row = &out.data[static_cast<std::size_t>(c) * cqt.cols];
        for (int bin = 0; bin < n_input; ++bin) {
            if (weight(c, bin) == 0.0f) continue;
            const float* src = &cqt.data[static_cast<std::size_t>(bin) * cqt.cols];
            for (int t = 0; t < cqt.cols; ++t) row[t] += src[t];
        }
    }

    for (int t = 0; t < out.cols; ++t) {
        float peak = 0.0f;
        for (int c = 0; c < n_chroma; ++c) peak = std::max(peak, std::abs(out.at(c, t)));
        const float denom = peak > 1e-20f ? peak : 1.0f;
        for (int c = 0; c < n_chroma; ++c) out.at(c, t) /= denom;
    }
    return out;
}

}  // namespace auvux::fastchroma::detail
