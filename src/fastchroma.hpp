// fastchroma — constant-Q transform and CQT chromagram.
#pragma once

#include <cstddef>
#include <cstdint>
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
[[nodiscard]] Matrix cqt(const float* y, std::size_t n, const CqtParams& p);

// Complex CQT, shape (n_bins, n_frames). Phase follows the e^{-i...} convention.
[[nodiscard]] ComplexMatrix cqt_complex(const float* y, std::size_t n, const CqtParams& p);

// CQT chromagram, shape (n_chroma, n_frames). Equal temperament; each frame is
// L-infinity normalized. fmin <= 0 selects C1.
[[nodiscard]] Matrix chroma(const float* y, std::size_t n, double sr, int hop,
                                int bins_per_octave, int n_octaves, int n_chroma, double fmin);

// GPU backend — vendor-neutral entry points. The backend compiled into the
// build implements them (Metal on Apple, CUDA on NVIDIA — backends are
// per-platform, so they never coexist); stubs on CPU-only builds. The compute
// functions return false when the backend is unavailable or the parameters
// are unsupported (caller falls back to the CPU path); on success the output
// matches the CPU result to float32 rounding (~1e-6 relative).
[[nodiscard]] bool gpu_available();
// Which backend this build carries: "metal", "cuda", or "none".
[[nodiscard]] const char* gpu_backend();
// output_mode: 0 magnitude, 1 power, 2 dB (10*log10(max(power, 1e-10))).
bool cqt_gpu(const float* y, std::size_t n, const CqtParams& p, Matrix& out,
             int output_mode = 0);
bool cqt_complex_gpu(const float* y, std::size_t n, const CqtParams& p, ComplexMatrix& out);
bool chroma_gpu(const float* y, std::size_t n, double sr, int hop, int bins_per_octave,
                    int n_octaves, int n_chroma, double fmin, Matrix& out);

// GPU-resident tensors (DLPack interop, as in fastmel). The handle is
// backend-specific — matching what PyTorch exchanges for the device: a raw
// device pointer on CUDA.
struct DeviceTensor {
    void* handle = nullptr;
    std::size_t byte_offset = 0;
};

// CQT of a device-resident signal of n samples, written to a freshly
// allocated device buffer of shape (p.n_bins, *n_frames_out): float32 for
// output_mode 0/1/2, interleaved complex64 for mode 3. The returned handle
// owns that buffer; release it with device_tensor_free. Unlike the
// bool-returning host entry points, throws std::runtime_error when the
// backend is missing or the parameters are unsupported (the caller decides
// whether to fall back to a host path).
//
// stream = 0 runs on an internal stream and blocks until complete.
// Non-zero is an external stream handle (CUDA: a cudaStream_t; 1 = the
// legacy default stream, per the DLPack convention): the work — including
// the output allocation — is enqueued on that stream and the call returns
// without host synchronization, like a native framework op. The caller
// must have made the input visible to that stream (DLPack
// __dlpack__(stream=...)).
DeviceTensor cqt_gpu_resident(DeviceTensor y, int n, const CqtParams& p, int output_mode,
                              std::uintptr_t stream, int* n_frames_out);

// CQT chromagram of a device-resident signal, shape (n_chroma, *n_frames_out).
DeviceTensor chroma_gpu_resident(DeviceTensor y, int n, double sr, int hop,
                                 int bins_per_octave, int n_octaves, int n_chroma,
                                 double fmin, std::uintptr_t stream, int* n_frames_out);

// DLPack device type of this build's GPU backend (kDLCUDA = 2,
// kDLMetal = 8); 0 for the stub.
int dlpack_device_type();

void device_tensor_free(void* handle) noexcept;

}  // namespace auvux::fastchroma
