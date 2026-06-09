// fastchroma — constant-Q transform and CQT chromagram.
#pragma once

#include <cstddef>
#include <vector>

namespace auvux::fastchroma {

struct CqtParams {
    double sr = 22050.0;
    int hop = 512;
    double fmin = 0.0;  // <= 0 selects C1 (32.70 Hz)
    int n_bins = 84;
    int bins_per_octave = 12;
    double filter_scale = 1.0;
    double sparsity = 0.01;
};

// Row-major (rows x cols) float matrix.
struct Matrix {
    std::vector<float> data;
    int rows = 0;
    int cols = 0;

    [[nodiscard]] float& at(int r, int c) { return data[static_cast<std::size_t>(r) * cols + c]; }
    [[nodiscard]] float at(int r, int c) const { return data[static_cast<std::size_t>(r) * cols + c]; }
};

// Row-major complex matrix (separate real/imag planes).
struct ComplexMatrix {
    std::vector<float> re, im;
    int rows = 0;
    int cols = 0;
};

// |CQT| magnitude, shape (n_bins, n_frames). Mono input.
[[nodiscard]] Matrix cqt_magnitude(const float* y, std::size_t n, const CqtParams& p);

// Complex CQT, shape (n_bins, n_frames). Phase follows the e^{-i...} convention.
[[nodiscard]] ComplexMatrix cqt_complex(const float* y, std::size_t n, const CqtParams& p);

// CQT chromagram, shape (n_chroma, n_frames). Equal temperament; each frame is
// L-infinity normalized. fmin <= 0 selects C1.
[[nodiscard]] Matrix chroma_cqt(const float* y, std::size_t n, double sr, int hop,
                                int bins_per_octave, int n_octaves, int n_chroma, double fmin);

}  // namespace auvux::fastchroma
