// CUDA (GPU) backend: the full CQT on one stream.
//
// A direct port of the Metal backend (see metal_backend.mm). The decimation
// pyramid runs as a chain of halfband FIR launches, then a fused kernel
// computes every (frame, octave) in one 2D grid: boxcar frame load -> radix-2
// FFT in shared memory -> real unpack -> sparse complex projection with the
// same SparseBasis values as the CPU path (built by the shared code in
// cqt_internal.hpp), so the two paths agree to float rounding. All octaves
// share one n_fft because bin frequencies and sample rate halve together.
// The FFT length is only known at engine-build time, so the kernel takes N
// and log2(N) as parameters and carves its buffers from dynamic shared
// memory.
//
// Engines (module constants + filterbank buffers + a private stream) are
// cached per parameter set and guarded by mutexes; callers may invoke
// concurrently without the GIL.
//
// Device-resident entry points (DLPack interop, as in fastmel) reuse the
// same engines: the input is copied device-to-device into the pyramid and
// the output is a stream-ordered allocation the caller owns. An engine
// event orders the shared pyramid across streams, so a resident call on an
// external stream (no host sync) cannot race a later call.
#include "cqt_internal.hpp"
#include "fastchroma.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// Oldest compute capability with SASS in the fatbin (set by CMakeLists to
// match the architecture list; the dev build script compiles -arch=native,
// where the default is always satisfied). PTX only JITs forward, so devices
// below this have no runnable kernel image.
#ifndef FASTCHROMA_MIN_CC
#define FASTCHROMA_MIN_CC 60
#endif

namespace auvux::fastchroma {
namespace {

using detail::SparseBasis;

constexpr int kMinCC = FASTCHROMA_MIN_CC;

__device__ inline float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

struct DecimParams {
    uint32_t in_off, in_n, out_off, out_n;
    float center, s2;
};

// out[m] = (center * x[2m] + sum_t taps[t] * x[2(m + t - 16) + 1]) * sqrt(2),
// one thread per output sample, reading/writing regions of the pyramid buffer.
__global__ void halfband(
    float* __restrict__ sig,
    const float* __restrict__ taps,   // 32 odd polyphase taps
    DecimParams d)
{
    const unsigned m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= d.out_n) return;
    const float* x = sig + d.in_off;
    const int pairs = int(d.in_n / 2);
    float acc = 0.0f;
    for (int t = 0; t < 32; ++t) {
        const int p = int(m) + t - 16;
        if (p >= 0 && p < pairs) acc += taps[t] * x[2 * p + 1];
    }
    sig[d.out_off + m] = (d.center * x[2 * m] + acc) * d.s2;
}

struct CqtKernelParams {
    uint32_t n_frames, mode;  // mode: 0 magnitude, 1 power, 2 dB, 3 complex
    uint32_t N, log2n;        // N = n_fft / 2
};

__global__ void cqt_kernel(
    const float* __restrict__ sig,       // decimation pyramid, concatenated
    const float2* __restrict__ tw,       // N/2: exp(-2pi i k / N)
    const float2* __restrict__ twr,      // N:   exp(-2pi i k / 2N)
    const int* __restrict__ sig_off,     // per octave
    const int* __restrict__ sig_len,
    const int* __restrict__ hops,
    const int* __restrict__ row_begin,
    const int* __restrict__ row_count,
    const int* __restrict__ row_start,   // CSR over all octaves' rows
    const int* __restrict__ col,
    const float2* __restrict__ val,      // pre-scaled by 1/sqrt(length)
    const int* __restrict__ out_bin,     // global row -> output bin
    float* __restrict__ out,             // (n_bins, n_frames); float2 for mode 3
    CqtKernelParams p)
{
    extern __shared__ float2 smem[];
    float2* buf = smem;            // N
    float2* spec = smem + p.N;     // N + 1
    const unsigned tid = threadIdx.x, tpb = blockDim.x;
    const unsigned frame = blockIdx.x, oc = blockIdx.y;
    const unsigned N = p.N;

    // Boxcar frame (the analysis window lives in the basis), centered, packed
    // into complex pairs in bit-reversed order.
    const int n = sig_len[oc];
    const int start = int(frame) * hops[oc] - int(N);  // pad = n_fft/2 = N
    const float* y = sig + sig_off[oc];
    for (unsigned i = tid; i < N; i += tpb) {
        const int i0 = start + int(2 * i), i1 = i0 + 1;
        const float a = (i0 >= 0 && i0 < n) ? y[i0] : 0.0f;
        const float b = (i1 >= 0 && i1 < n) ? y[i1] : 0.0f;
        buf[__brev(i) >> (32 - p.log2n)] = make_float2(a, b);
    }
    __syncthreads();

    // Radix-2 DIT, one butterfly per thread per stage (tpb == N/2).
    for (unsigned s = 0; s < p.log2n; ++s) {
        for (unsigned b = tid; b < N / 2; b += tpb) {
            const unsigned q = 1u << s;
            const unsigned j = b & (q - 1);
            const unsigned i0 = ((b >> s) << (s + 1)) + j;
            const float2 a = buf[i0];
            const float2 c = cmul(tw[j * (N >> (s + 1))], buf[i0 + q]);
            buf[i0] = make_float2(a.x + c.x, a.y + c.y);
            buf[i0 + q] = make_float2(a.x - c.x, a.y - c.y);
        }
        __syncthreads();
    }

    // Real-input unpack: spec[0..N] is the standard half spectrum, matching
    // the CPU RealFFT.
    for (unsigned k = tid; k < N; k += tpb) {
        const unsigned kc = (N - k) & (N - 1);
        const float2 zk = buf[k], zc = buf[kc];
        const float2 fe = make_float2(0.5f * (zk.x + zc.x), 0.5f * (zk.y - zc.y));
        const float2 fo = make_float2(0.5f * (zk.y + zc.y), 0.5f * (zc.x - zk.x));
        const float2 t = cmul(twr[k], fo);
        spec[k] = make_float2(fe.x + t.x, fe.y + t.y);
    }
    if (tid == 0) spec[N] = make_float2(buf[0].x - buf[0].y, 0.0f);
    __syncthreads();

    // Sparse complex projection; one thread per row of this octave's basis.
    // The response is complex throughout; p.mode only selects the final store.
    const unsigned rb = unsigned(row_begin[oc]), re = rb + unsigned(row_count[oc]);
    for (unsigned r = rb + tid; r < re; r += tpb) {
        float ar = 0.0f, ai = 0.0f;
        for (int idx = row_start[r]; idx < row_start[r + 1]; ++idx) {
            const float2 v = val[idx];
            const float2 x = spec[col[idx]];
            ar += v.x * x.x - v.y * x.y;
            ai += v.x * x.y + v.y * x.x;
        }
        const unsigned o = unsigned(out_bin[r]) * p.n_frames + frame;
        const float pw = ar * ar + ai * ai;
        if (p.mode == 0u)      out[o] = sqrtf(pw);
        else if (p.mode == 1u) out[o] = pw;
        else if (p.mode == 2u) out[o] = 10.0f * log10f(fmaxf(pw, 1e-10f));
        else                   reinterpret_cast<float2*>(out)[o] = make_float2(ar, ai);
    }
}

// Chroma fold for the device-resident path, mirroring detail::chroma_fold:
// each CQT bin maps to exactly one pitch class, derived from scalars, so the
// kernel needs no tables. Bins are accumulated in ascending order — the same
// summation order as the CPU fold.
struct FoldParams {
    uint32_t n_bins, n_frames, bpo, n_chroma, merge, roll_cols, roll;
};

__global__ void chroma_fold_kernel(
    const float* __restrict__ cqt,   // (n_bins, n_frames) magnitudes
    float* __restrict__ out,         // (n_chroma, n_frames)
    FoldParams f)
{
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned c = blockIdx.y;
    if (t >= f.n_frames) return;
    const unsigned src = (c + f.n_chroma - f.roll) % f.n_chroma;
    float acc = 0.0f;
    for (unsigned bin = 0; bin < f.n_bins; ++bin) {
        const unsigned folded = (bin % f.bpo + f.roll_cols) % f.bpo;
        if (folded / f.merge == src) acc += cqt[bin * f.n_frames + t];
    }
    out[c * f.n_frames + t] = acc;
}

// Per-frame L-infinity normalization, one thread per frame.
__global__ void chroma_norm_kernel(float* __restrict__ out, uint32_t n_chroma,
                                   uint32_t n_frames)
{
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_frames) return;
    float peak = 0.0f;
    for (unsigned c = 0; c < n_chroma; ++c)
        peak = fmaxf(peak, fabsf(out[c * n_frames + t]));
    const float denom = peak > 1e-20f ? peak : 1.0f;
    for (unsigned c = 0; c < n_chroma; ++c) out[c * n_frames + t] /= denom;
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error (") + what + "): " +
                                 cudaGetErrorString(e));
}

// Output buffers for resident calls come from the stream-ordered pool
// allocator when the driver supports it (a fresh cudaMalloc costs more than
// the kernels); pool-allocated pointers must be freed with cudaFreeAsync, so
// remember which allocator was used process-wide.
std::atomic<int>& mempools_supported() {
    static std::atomic<int> supported{-1};
    return supported;
}

float* alloc_output(std::size_t bytes, cudaStream_t st) {
    if (mempools_supported().load() < 0) {
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMemoryPoolsSupported, 0) != cudaSuccess)
            v = 0;
        mempools_supported().store(v);
    }
    float* p = nullptr;
    if (mempools_supported().load())
        check(cudaMallocAsync(&p, bytes, st), "output buffer");
    else
        check(cudaMalloc(&p, bytes), "output buffer");
    return p;
}

void free_output(float* p, cudaStream_t st) noexcept {
    if (!p) return;
    if (mempools_supported().load() > 0)
        cudaFreeAsync(p, st);
    else
        cudaFree(p);  // implicitly synchronizes; pre-mempool drivers only
}

// Owning device allocation; uploads are typed by the element the kernel reads.
struct DevBuf {
    void* p = nullptr;
    ~DevBuf() { reset(); }
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;

    void reset() {
        if (p) cudaFree(p);
        p = nullptr;
    }
    void alloc(std::size_t bytes, const char* what) {
        reset();
        check(cudaMalloc(&p, bytes), what);
    }
    template <typename T>
    void upload(const std::vector<T>& v, const char* what) {
        alloc(v.size() * sizeof(T), what);
        check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }
    template <typename T>
    T* get() const { return static_cast<T*>(p); }
};

// One transform configuration: device constants, filterbank buffers, a
// private stream, and (resized on demand) the pyramid/output buffers. Not
// internally synchronized — the cache below wraps each engine in a mutex.
class GpuCqt {
public:
    explicit GpuCqt(const CqtParams& p) : p_{p} {
        if (p_.fmin <= 0.0) p_.fmin = detail::kC1Hz;
        n_octaves_ = static_cast<int>(std::ceil(static_cast<double>(p_.n_bins) / p_.bins_per_octave));
        const int n_filters = std::min(p_.bins_per_octave, p_.n_bins);

        // Every octave-to-octave step must decimate (the shared n_fft depends
        // on it), so the hop needs a factor of two per step.
        if (detail::two_factor_count(p_.hop) < n_octaves_ - 1)
            throw std::runtime_error("cuda: hop must be divisible by 2^(n_octaves-1)");

        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "device query");
        if (prop.major * 10 + prop.minor < kMinCC)
            throw std::runtime_error(
                "GPU compute capability " + std::to_string(prop.major) + "." +
                std::to_string(prop.minor) + " is below " +
                std::to_string(kMinCC / 10) + "." + std::to_string(kMinCC % 10) +
                ", the minimum this fastchroma build supports");

        const auto freqs = detail::cqt_frequencies(p_.n_bins, p_.fmin, p_.bins_per_octave);
        const auto alpha = detail::relative_bandwidth(freqs);
        double cutoff = 0.0;
        detail::wavelet_lengths(freqs, p_.sr, alpha, p_.filter_scale, &cutoff);

        // Optional upfront downsampling when the top octave leaves headroom;
        // mirrors cqt_core. The pyramid simply gains `pre_` extra levels in
        // front of octave 0.
        const int down1 = std::max(
            0, static_cast<int>(std::ceil(std::log2(p_.sr / 2.0 / cutoff))) - 2);
        const int down2 = std::max(0, detail::two_factor_count(p_.hop) - n_octaves_ + 1);
        pre_ = std::min(down1, down2);
        base_sr_ = p_.sr / std::pow(2.0, pre_);
        hop0_ = p_.hop >> pre_;
        n_levels_ = pre_ + n_octaves_;

        const auto lengths = detail::wavelet_lengths(freqs, base_sr_, alpha, p_.filter_scale);

        // Per-octave bases at halved rates; all share one n_fft because the
        // bin frequencies and the sample rate halve together.
        std::vector<SparseBasis> bases(n_octaves_);
        for (int oc = 0; oc < n_octaves_; ++oc) {
            const int lo = std::max(0, p_.n_bins - n_filters * (oc + 1));
            const int hi = (oc == 0) ? p_.n_bins : p_.n_bins - n_filters * oc;
            const std::vector<double> of(freqs.begin() + lo, freqs.begin() + hi);
            const std::vector<double> oa(alpha.begin() + lo, alpha.begin() + hi);
            bases[oc] = detail::build_basis(of, oa, base_sr_ / std::pow(2.0, oc), base_sr_,
                                            p_.filter_scale, p_.sparsity);
            if (oc == 0) n_fft_ = bases[oc].n_fft;
            if (bases[oc].n_fft != n_fft_)
                throw std::runtime_error("cuda: octaves disagree on n_fft");
        }
        N_ = n_fft_ / 2;
        log2n_ = fft::ilog2(N_);
        tpb_ = N_ / 2;
        shmem_ = static_cast<std::size_t>(2 * N_ + 1) * sizeof(float2);
        if (tpb_ > prop.maxThreadsPerBlock ||
            shmem_ > static_cast<std::size_t>(prop.sharedMemPerBlock))
            throw std::runtime_error("cuda: n_fft too large for this device");

        // Flatten the CSR bases; fold the final 1/sqrt(length) row scaling
        // (cqt_core's last loop) into the values.
        std::vector<int> row_begin(n_octaves_), row_count(n_octaves_), row_start{0}, col, out_bin;
        std::vector<float2> val;
        for (int oc = 0; oc < n_octaves_; ++oc) {
            const SparseBasis& b = bases[oc];
            const int lo = std::max(0, p_.n_bins - n_filters * (oc + 1));
            row_begin[oc] = static_cast<int>(row_start.size()) - 1;
            row_count[oc] = b.rows;
            for (int r = 0; r < b.rows; ++r) {
                const float s = static_cast<float>(1.0 / std::sqrt(lengths[lo + r]));
                for (int idx = b.row_start[r]; idx < b.row_start[r + 1]; ++idx) {
                    col.push_back(b.col[idx]);
                    val.push_back(make_float2(b.val[idx].real() * s, b.val[idx].imag() * s));
                }
                row_start.push_back(static_cast<int>(col.size()));
                out_bin.push_back(lo + r);
            }
        }

        // Twiddles: double-precision trig, stored float.
        std::vector<float2> tw_table(N_ / 2), twr_table(N_);
        for (int j = 0; j < N_ / 2; ++j)
            tw_table[j] = make_float2(static_cast<float>(std::cos(-2.0 * detail::kPi * j / N_)),
                                      static_cast<float>(std::sin(-2.0 * detail::kPi * j / N_)));
        for (int k = 0; k < N_; ++k)
            twr_table[k] = make_float2(static_cast<float>(std::cos(-2.0 * detail::kPi * k / n_fft_)),
                                       static_cast<float>(std::sin(-2.0 * detail::kPi * k / n_fft_)));

        check(cudaStreamCreate(&stream_), "stream");
        check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming), "event");
        tw_.upload(tw_table, "twiddles");
        twr_.upload(twr_table, "real twiddles");
        rb_.upload(row_begin, "row begins");
        rc_.upload(row_count, "row counts");
        rs_.upload(row_start, "row starts");
        col_.upload(col, "columns");
        val_.upload(val, "values");
        ob_.upload(out_bin, "output bins");
        const std::vector<float> taps(hb_.odd.begin(), hb_.odd.end());
        taps_.upload(taps, "halfband taps");
    }

    ~GpuCqt() {
        if (done_) cudaEventDestroy(done_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    GpuCqt(const GpuCqt&) = delete;
    GpuCqt& operator=(const GpuCqt&) = delete;

    // Runs the transform on a host signal. mode 0/1/2 writes float
    // magnitudes/power/dB into `outf`; mode 3 writes the complex response
    // into `outc`.
    void compute(const float* y, int n, int mode, Matrix* outf, ComplexMatrix* outc) {
        ensure_capacity(n);
        check(cudaStreamWaitEvent(stream_, done_, 0), "event wait");
        check(cudaMemcpyAsync(sig_.get<float>(), y, static_cast<std::size_t>(n) * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), "input copy");
        enqueue(stream_, mode, out_.get<float>());
        check(cudaEventRecord(done_, stream_), "event record");

        const std::size_t count = static_cast<std::size_t>(p_.n_bins) * n_frames_;
        if (mode == 3) {
            std::vector<float> result(2 * count);
            check(cudaMemcpyAsync(result.data(), out_.p, 2 * count * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_), "output copy");
            check(cudaStreamSynchronize(stream_), "stream sync");
            outc->rows = p_.n_bins;
            outc->cols = n_frames_;
            outc->re.resize(count);
            outc->im.resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                outc->re[i] = result[2 * i];
                outc->im[i] = result[2 * i + 1];
            }
        } else {
            outf->rows = p_.n_bins;
            outf->cols = n_frames_;
            outf->data.resize(count);
            check(cudaMemcpyAsync(outf->data.data(), out_.p, count * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_), "output copy");
            check(cudaStreamSynchronize(stream_), "stream sync");
        }
    }

    // Device-resident transform: D2D the input into the pyramid, run, and
    // return a freshly allocated device output (float32, or interleaved
    // complex64 for mode 3). External streams skip the host sync.
    float* compute_resident(DeviceTensor y, int n, int mode, std::uintptr_t stream) {
        ensure_capacity(n);
        const cudaStream_t st = stream ? reinterpret_cast<cudaStream_t>(stream) : stream_;
        check(cudaStreamWaitEvent(st, done_, 0), "event wait");
        stage_input(y, n, st);
        const std::size_t floats =
            (mode == 3 ? 2u : 1u) * static_cast<std::size_t>(p_.n_bins) * n_frames_;
        float* dev_out = alloc_output(floats * sizeof(float), st);
        try {
            enqueue(st, mode, dev_out);
            check(cudaEventRecord(done_, st), "event record");
            if (!stream) check(cudaStreamSynchronize(stream_), "stream sync");
        } catch (...) {
            free_output(dev_out, st);
            throw;
        }
        return dev_out;
    }

    // Device-resident chromagram: magnitude CQT into a stream-ordered scratch
    // buffer, then the fold + per-frame normalization, all on `st`.
    float* compute_chroma_resident(DeviceTensor y, int n, int n_chroma, double f0,
                                   std::uintptr_t stream) {
        ensure_capacity(n);
        const cudaStream_t st = stream ? reinterpret_cast<cudaStream_t>(stream) : stream_;
        check(cudaStreamWaitEvent(st, done_, 0), "event wait");
        stage_input(y, n, st);

        const int merge = p_.bins_per_octave / n_chroma;
        if (merge < 1) throw std::runtime_error("cuda: n_chroma exceeds bins_per_octave");
        const double midi0 =
            std::fmod(12.0 * (std::log2(f0) - std::log2(440.0)) + 69.0, 12.0);
        int roll = static_cast<int>(std::lround(midi0 * (n_chroma / 12.0)));
        roll = ((roll % n_chroma) + n_chroma) % n_chroma;

        float* mag = alloc_output(
            static_cast<std::size_t>(p_.n_bins) * n_frames_ * sizeof(float), st);
        float* dev_out = nullptr;
        try {
            enqueue(st, 0, mag);
            dev_out = alloc_output(
                static_cast<std::size_t>(n_chroma) * n_frames_ * sizeof(float), st);
            const FoldParams f{static_cast<uint32_t>(p_.n_bins),
                               static_cast<uint32_t>(n_frames_),
                               static_cast<uint32_t>(p_.bins_per_octave),
                               static_cast<uint32_t>(n_chroma),
                               static_cast<uint32_t>(merge),
                               static_cast<uint32_t>(merge / 2),
                               static_cast<uint32_t>(roll)};
            const unsigned blocks = (static_cast<unsigned>(n_frames_) + 255u) / 256u;
            chroma_fold_kernel<<<dim3(blocks, n_chroma), 256, 0, st>>>(mag, dev_out, f);
            chroma_norm_kernel<<<blocks, 256, 0, st>>>(
                dev_out, static_cast<uint32_t>(n_chroma), static_cast<uint32_t>(n_frames_));
            check(cudaGetLastError(), "kernel launch");
            check(cudaEventRecord(done_, st), "event record");
            free_output(mag, st);
            mag = nullptr;
            if (!stream) check(cudaStreamSynchronize(stream_), "stream sync");
        } catch (...) {
            free_output(mag, st);
            free_output(dev_out, st);
            throw;
        }
        return dev_out;
    }

    int n_frames() const { return n_frames_; }

private:
    // Decimation pyramid + the fused CQT kernel writing to `out`, all
    // enqueued on `st`. The caller stages the input and orders the stream.
    void enqueue(cudaStream_t st, int mode, float* out) {
        for (int lvl = 1; lvl < n_levels_; ++lvl) {
            const DecimParams d{
                static_cast<uint32_t>(lvl_off_[lvl - 1]), static_cast<uint32_t>(lvl_len_[lvl - 1]),
                static_cast<uint32_t>(lvl_off_[lvl]), static_cast<uint32_t>(lvl_len_[lvl]),
                hb_.center, 1.4142135623730951f};
            const unsigned blocks = (static_cast<unsigned>(lvl_len_[lvl]) + 255u) / 256u;
            halfband<<<blocks, 256, 0, st>>>(sig_.get<float>(), taps_.get<float>(), d);
        }
        const CqtKernelParams params{static_cast<uint32_t>(n_frames_), static_cast<uint32_t>(mode),
                                     static_cast<uint32_t>(N_), static_cast<uint32_t>(log2n_)};
        cqt_kernel<<<dim3(n_frames_, n_octaves_), tpb_, shmem_, st>>>(
            sig_.get<float>(), tw_.get<float2>(), twr_.get<float2>(), so_.get<int>(),
            sl_.get<int>(), h_.get<int>(), rb_.get<int>(), rc_.get<int>(), rs_.get<int>(),
            col_.get<int>(), val_.get<float2>(), ob_.get<int>(), out, params);
        check(cudaGetLastError(), "kernel launch");
    }

    void stage_input(DeviceTensor y, int n, cudaStream_t st) {
        const auto* dev_y = reinterpret_cast<const float*>(
            static_cast<const char*>(y.handle) + y.byte_offset);
        check(cudaMemcpyAsync(sig_.get<float>(), dev_y,
                              static_cast<std::size_t>(n) * sizeof(float),
                              cudaMemcpyDeviceToDevice, st), "input copy");
    }
    void ensure_capacity(int n) {
        if (n == cur_n_) return;
        // A resident call on an external stream may still be running against
        // the old buffers; wait for it before freeing them.
        if (done_) check(cudaEventSynchronize(done_), "event sync");
        cur_n_ = -1;  // invalid until every allocation below succeeds
        lvl_off_.resize(n_levels_);
        lvl_len_.resize(n_levels_);
        int total = 0;
        for (int lvl = 0, len = n; lvl < n_levels_; ++lvl, len = (len + 1) / 2) {
            lvl_off_[lvl] = total;
            lvl_len_[lvl] = len;
            total += len;
        }
        // Octave oc reads pyramid level pre_ + oc.
        std::vector<int> sig_off(n_octaves_), sig_len(n_octaves_), hops(n_octaves_);
        n_frames_ = std::numeric_limits<int>::max();
        for (int oc = 0; oc < n_octaves_; ++oc) {
            sig_off[oc] = lvl_off_[pre_ + oc];
            sig_len[oc] = lvl_len_[pre_ + oc];
            hops[oc] = hop0_ >> oc;
            n_frames_ = std::min(n_frames_, 1 + sig_len[oc] / hops[oc]);
        }

        sig_.alloc(static_cast<std::size_t>(total) * sizeof(float), "pyramid buffer");
        so_.upload(sig_off, "signal offsets");
        sl_.upload(sig_len, "signal lengths");
        h_.upload(hops, "hops");
        // Sized for the largest output (complex: two floats per element).
        out_.alloc(static_cast<std::size_t>(p_.n_bins) * n_frames_ * 2 * sizeof(float),
                   "output buffer");
        cur_n_ = n;
    }

    CqtParams p_;
    int n_octaves_ = 0, n_fft_ = 0, N_ = 0, log2n_ = 0, tpb_ = 0;
    int pre_ = 0, n_levels_ = 0, hop0_ = 0;
    double base_sr_ = 0.0;
    std::size_t shmem_ = 0;
    detail::HalfBand hb_;

    cudaStream_t stream_ = nullptr;
    cudaEvent_t done_ = nullptr;  // orders the shared buffers across streams
    DevBuf tw_, twr_, rb_, rc_, rs_, col_, val_, ob_, taps_;

    // Sized by ensure_capacity().
    int cur_n_ = -1, n_frames_ = 0;
    std::vector<int> lvl_off_, lvl_len_;
    DevBuf sig_, so_, sl_, h_, out_;
};

struct Engine {
    explicit Engine(const CqtParams& p) : gpu{p} {}
    GpuCqt gpu;
    std::mutex mutex;
};

// Engines are cached per parameter set and never evicted, so references stay
// valid after the cache lock is released.
Engine& engine_for(const CqtParams& p) {
    using Key = std::tuple<double, int, double, int, int, double, double>;
    static std::map<Key, std::unique_ptr<Engine>> cache;
    static std::mutex cache_mutex;
    const Key key{p.sr, p.hop, p.fmin, p.n_bins, p.bins_per_octave, p.filter_scale, p.sparsity};
    const std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, std::make_unique<Engine>(p)).first;
    return *it->second;
}

}  // namespace

// "Available" means usable by this build: present and new enough to run
// the embedded SASS/PTX.
bool gpu_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return false;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
    return prop.major * 10 + prop.minor >= kMinCC;
}

const char* gpu_backend() { return "cuda"; }

static bool run_gpu(const float* y, std::size_t n, const CqtParams& p, int mode, Matrix* outf,
                    ComplexMatrix* outc) {
    if (!gpu_available() || n < 1 || n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return false;
    try {
        Engine& eng = engine_for(p);
        const std::lock_guard<std::mutex> lock(eng.mutex);
        eng.gpu.compute(y, static_cast<int>(n), mode, outf, outc);
        return true;
    } catch (...) {
        return false;  // unsupported parameters or a device failure: CPU fallback
    }
}

bool cqt_gpu(const float* y, std::size_t n, const CqtParams& p, Matrix& out, int output_mode) {
    if (output_mode < 0 || output_mode > 2) return false;
    return run_gpu(y, n, p, output_mode, &out, nullptr);
}

bool cqt_complex_gpu(const float* y, std::size_t n, const CqtParams& p, ComplexMatrix& out) {
    return run_gpu(y, n, p, 3, nullptr, &out);
}

bool chroma_gpu(const float* y, std::size_t n, double sr, int hop, int bins_per_octave,
                    int n_octaves, int n_chroma, double fmin, Matrix& out) {
    const double f0 = (fmin > 0.0) ? fmin : detail::kC1Hz;
    CqtParams p;
    p.sr = sr;
    p.hop = hop;
    p.fmin = f0;
    p.n_bins = n_octaves * bins_per_octave;
    p.bins_per_octave = bins_per_octave;
    Matrix mag;
    if (!cqt_gpu(y, n, p, mag)) return false;
    out = detail::chroma_fold(mag, bins_per_octave, n_chroma, f0);
    return true;
}

DeviceTensor cqt_gpu_resident(DeviceTensor y, int n, const CqtParams& p, int output_mode,
                              std::uintptr_t stream, int* n_frames_out) {
    if (output_mode < 0 || output_mode > 3)
        throw std::invalid_argument("output_mode must be 0..3");
    if (n < 1) throw std::invalid_argument("y must not be empty");
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        throw std::runtime_error("no CUDA device available");

    Engine& eng = engine_for(p);
    const std::lock_guard<std::mutex> lock(eng.mutex);
    float* out = eng.gpu.compute_resident(y, n, output_mode, stream);
    *n_frames_out = eng.gpu.n_frames();
    return {out, 0};
}

DeviceTensor chroma_gpu_resident(DeviceTensor y, int n, double sr, int hop,
                                 int bins_per_octave, int n_octaves, int n_chroma,
                                 double fmin, std::uintptr_t stream, int* n_frames_out) {
    if (n < 1) throw std::invalid_argument("y must not be empty");
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        throw std::runtime_error("no CUDA device available");

    const double f0 = (fmin > 0.0) ? fmin : detail::kC1Hz;
    CqtParams p;
    p.sr = sr;
    p.hop = hop;
    p.fmin = f0;
    p.n_bins = n_octaves * bins_per_octave;
    p.bins_per_octave = bins_per_octave;
    Engine& eng = engine_for(p);
    const std::lock_guard<std::mutex> lock(eng.mutex);
    float* out = eng.gpu.compute_chroma_resident(y, n, n_chroma, f0, stream);
    *n_frames_out = eng.gpu.n_frames();
    return {out, 0};
}

int dlpack_device_type() { return 2; }  // kDLCUDA

void device_tensor_free(void* handle) noexcept {
    if (!handle) return;
    if (mempools_supported().load() > 0)
        cudaFreeAsync(handle, nullptr);  // legacy default stream
    else
        cudaFree(handle);
}

}  // namespace auvux::fastchroma
