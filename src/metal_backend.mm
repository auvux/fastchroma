// Metal (GPU) backend: the full CQT in one command buffer.
//
// The decimation pyramid runs as a chain of halfband FIR dispatches, then a
// fused kernel computes every (frame, octave) in one 2D grid: boxcar frame
// load -> radix-2 FFT in threadgroup memory -> real unpack -> sparse complex
// projection with the same SparseBasis values as the CPU path (built by the
// shared code in cqt_internal.hpp), so the two paths agree to float rounding.
// All octaves share one n_fft because bin frequencies and sample rate halve
// together. Kernels are compiled at runtime from source — no metallib in the
// wheel, no Metal toolchain at build time.
//
// Engines (pipelines + filterbank buffers) are cached per parameter set and
// guarded by mutexes; callers may invoke concurrently without the GIL.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "cqt_internal.hpp"
#include "fastchroma.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace auvux::fastchroma {
namespace {

using detail::SparseBasis;

const char* kernel_body() {
    return R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params { uint n_frames; uint mode; };  // 0 magnitude, 1 power, 2 dB, 3 complex

struct DecimParams {
    uint in_off, in_n, out_off, out_n;
    float center, s2;
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// out[m] = (center * x[2m] + sum_t taps[t] * x[2(m + t - 16) + 1]) * sqrt(2),
// one thread per output sample, reading/writing regions of the pyramid buffer.
kernel void halfband(
    device float*         sig  [[buffer(0)]],
    device const float*   taps [[buffer(1)]],   // 32 odd polyphase taps
    constant DecimParams& d    [[buffer(2)]],
    uint m [[thread_position_in_grid]])
{
    if (m >= d.out_n) return;
    device const float* x = sig + d.in_off;
    const int pairs = int(d.in_n / 2);
    float acc = 0.0f;
    for (int t = 0; t < 32; ++t) {
        const int p = int(m) + t - 16;
        if (p >= 0 && p < pairs) acc += taps[t] * x[2 * p + 1];
    }
    sig[d.out_off + m] = (d.center * x[2 * m] + acc) * d.s2;
}

kernel void cqt(
    device const float*  sig       [[buffer(0)]],   // decimation pyramid, concatenated
    device const float2* tw        [[buffer(1)]],   // N/2: exp(-2pi i k / N)
    device const float2* twr       [[buffer(2)]],   // N:   exp(-2pi i k / 2N)
    device const int*    sig_off   [[buffer(3)]],   // per octave
    device const int*    sig_len   [[buffer(4)]],
    device const int*    hops      [[buffer(5)]],
    device const int*    row_begin [[buffer(6)]],
    device const int*    row_count [[buffer(7)]],
    device const int*    row_start [[buffer(8)]],   // CSR over all octaves' rows
    device const int*    col       [[buffer(9)]],
    device const float2* val       [[buffer(10)]],  // pre-scaled by 1/sqrt(length)
    device const int*    out_bin   [[buffer(11)]],  // global row -> output bin
    device float*        out       [[buffer(12)]],  // (n_bins, n_frames), modes 0-2
    constant Params&     p         [[buffer(13)]],
    device float2*       outc      [[buffer(14)]],  // same buffer as float2, mode 3
    uint tid [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    const uint frame = tg.x, oc = tg.y;
    threadgroup float2 buf[N];
    threadgroup float2 spec[N + 1];

    // Boxcar frame (the analysis window lives in the basis), centered, packed
    // into complex pairs in bit-reversed order.
    const int n = sig_len[oc];
    const int start = int(frame) * hops[oc] - int(N);  // pad = n_fft/2 = N
    device const float* y = sig + sig_off[oc];
    for (uint j = 0; j < N / TPB; ++j) {
        const uint i = tid + j * TPB;
        const int i0 = start + int(2 * i), i1 = i0 + 1;
        const float a = (i0 >= 0 && i0 < n) ? y[i0] : 0.0f;
        const float b = (i1 >= 0 && i1 < n) ? y[i1] : 0.0f;
        buf[reverse_bits(i) >> (32 - LOG2N)] = float2(a, b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Radix-2 DIT, one butterfly per thread per stage (TPB == N/2).
    for (uint s = 0; s < LOG2N; ++s) {
        for (uint b = tid; b < N / 2; b += TPB) {
            const uint q = 1u << s;
            const uint j = b & (q - 1);
            const uint i0 = ((b >> s) << (s + 1)) + j;
            const float2 a = buf[i0];
            const float2 c = cmul(tw[j * (N >> (s + 1))], buf[i0 + q]);
            buf[i0] = a + c;
            buf[i0 + q] = a - c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Real-input unpack: spec[0..N] is the standard half spectrum, matching the
    // CPU RealFFT (vDSP zrip halved).
    for (uint c = 0; c < N / TPB; ++c) {
        const uint k = tid + c * TPB;
        const uint kc = (N - k) & (N - 1);
        const float2 zk = buf[k], zc = buf[kc];
        const float2 fe = 0.5f * float2(zk.x + zc.x, zk.y - zc.y);
        const float2 fo = 0.5f * float2(zk.y + zc.y, zc.x - zk.x);
        spec[k] = fe + cmul(twr[k], fo);
    }
    if (tid == 0) spec[N] = float2(buf[0].x - buf[0].y, 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sparse complex projection; one thread per row of this octave's basis.
    // The response is complex throughout; p.mode only selects the final store.
    const uint rb = uint(row_begin[oc]), re = rb + uint(row_count[oc]);
    for (uint r = rb + tid; r < re; r += TPB) {
        float ar = 0.0f, ai = 0.0f;
        for (int idx = row_start[r]; idx < row_start[r + 1]; ++idx) {
            const float2 v = val[idx];
            const float2 x = spec[col[idx]];
            ar += v.x * x.x - v.y * x.y;
            ai += v.x * x.y + v.y * x.x;
        }
        const uint o = uint(out_bin[r]) * p.n_frames + frame;
        const float pw = ar * ar + ai * ai;
        if (p.mode == 0u)      out[o] = sqrt(pw);
        else if (p.mode == 1u) out[o] = pw;
        else if (p.mode == 2u) out[o] = 10.0f * log10(max(pw, 1e-10f));
        else                   outc[o] = float2(ar, ai);
    }
}
)METAL";
}

id<MTLDevice> shared_device() {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}

// One transform configuration: pipelines, filterbank buffers, and (resized on
// demand) the pyramid/output buffers. Not internally synchronized — the cache
// below wraps each engine in a mutex.
class GpuCqt {
public:
    explicit GpuCqt(const CqtParams& p) : p_{p} {
        if (p_.fmin <= 0.0) p_.fmin = detail::kC1Hz;
        n_octaves_ = static_cast<int>(std::ceil(static_cast<double>(p_.n_bins) / p_.bins_per_octave));
        const int n_filters = std::min(p_.bins_per_octave, p_.n_bins);

        // Every octave-to-octave step must decimate (the shared n_fft depends
        // on it), so the hop needs a factor of two per step.
        if (detail::two_factor_count(p_.hop) < n_octaves_ - 1)
            throw std::runtime_error("metal: hop must be divisible by 2^(n_octaves-1)");

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
                throw std::runtime_error("metal: octaves disagree on n_fft");
        }
        N_ = n_fft_ / 2;
        log2n_ = fft::ilog2(N_);
        tpb_ = N_ / 2;

        // Flatten the CSR bases; fold the final 1/sqrt(length) row scaling
        // (cqt_core's last loop) into the values.
        std::vector<int> row_begin(n_octaves_), row_count(n_octaves_), row_start{0}, col, out_bin;
        std::vector<float> val;
        for (int oc = 0; oc < n_octaves_; ++oc) {
            const SparseBasis& b = bases[oc];
            const int lo = std::max(0, p_.n_bins - n_filters * (oc + 1));
            row_begin[oc] = static_cast<int>(row_start.size()) - 1;
            row_count[oc] = b.rows;
            for (int r = 0; r < b.rows; ++r) {
                const float s = static_cast<float>(1.0 / std::sqrt(lengths[lo + r]));
                for (int idx = b.row_start[r]; idx < b.row_start[r + 1]; ++idx) {
                    col.push_back(b.col[idx]);
                    val.push_back(b.val[idx].real() * s);
                    val.push_back(b.val[idx].imag() * s);
                }
                row_start.push_back(static_cast<int>(col.size()));
                out_bin.push_back(lo + r);
            }
        }

        std::vector<float> tw_table(N_), twr_table(2 * N_);
        for (int j = 0; j < N_ / 2; ++j) {
            tw_table[2 * j] = static_cast<float>(std::cos(-2.0 * detail::kPi * j / N_));
            tw_table[2 * j + 1] = static_cast<float>(std::sin(-2.0 * detail::kPi * j / N_));
        }
        for (int k = 0; k < N_; ++k) {
            twr_table[2 * k] = static_cast<float>(std::cos(-2.0 * detail::kPi * k / n_fft_));
            twr_table[2 * k + 1] = static_cast<float>(std::sin(-2.0 * detail::kPi * k / n_fft_));
        }

        dev_ = shared_device();
        if (!dev_) throw std::runtime_error("metal: no device");
        char header[128];
        std::snprintf(header, sizeof(header),
                      "constant uint N = %d;\nconstant uint LOG2N = %d;\nconstant uint TPB = %d;\n",
                      N_, log2n_, tpb_);
        NSError* err = nil;
        id<MTLLibrary> lib =
            [dev_ newLibraryWithSource:[NSString stringWithFormat:@"%s%s", header, kernel_body()]
                               options:nil
                                 error:&err];
        if (!lib) throw std::runtime_error("metal: kernel compile failed");
        psoCqt_ = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cqt"]
                                                      error:&err];
        psoDecim_ = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"halfband"]
                                                        error:&err];
        if (!psoCqt_ || !psoDecim_) throw std::runtime_error("metal: pipeline failed");
        if (static_cast<NSUInteger>(tpb_) > psoCqt_.maxTotalThreadsPerThreadgroup)
            throw std::runtime_error("metal: threadgroup too large for device");
        queue_ = [dev_ newCommandQueue];

        const auto mkbuf = [&](const void* ptr, std::size_t bytes) {
            return [dev_ newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
        };
        twBuf_ = mkbuf(tw_table.data(), tw_table.size() * sizeof(float));
        twrBuf_ = mkbuf(twr_table.data(), twr_table.size() * sizeof(float));
        rbBuf_ = mkbuf(row_begin.data(), row_begin.size() * sizeof(int));
        rcBuf_ = mkbuf(row_count.data(), row_count.size() * sizeof(int));
        rsBuf_ = mkbuf(row_start.data(), row_start.size() * sizeof(int));
        colBuf_ = mkbuf(col.data(), col.size() * sizeof(int));
        valBuf_ = mkbuf(val.data(), val.size() * sizeof(float));
        obBuf_ = mkbuf(out_bin.data(), out_bin.size() * sizeof(int));
        tapsBuf_ = mkbuf(hb_.odd.data(), hb_.odd.size() * sizeof(float));
    }

    ~GpuCqt() { std::free(mem_); }
    GpuCqt(const GpuCqt&) = delete;
    GpuCqt& operator=(const GpuCqt&) = delete;

    // Runs the transform. mode 0/1/2 writes float magnitudes/power/dB into
    // `outf`; mode 3 writes the complex response into `outc`.
    void compute(const float* y, int n, int mode, Matrix* outf, ComplexMatrix* outc) {
        ensure_capacity(n);
        std::memcpy(mem_, y, static_cast<std::size_t>(n) * sizeof(float));

        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            // The encoder's default serial dispatch keeps the pyramid levels
            // and the CQT ordered with memory coherence between dispatches.
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int lvl = 1; lvl < n_levels_; ++lvl) {
                const struct { uint32_t in_off, in_n, out_off, out_n; float center, s2; } d{
                    static_cast<uint32_t>(lvl_off_[lvl - 1]), static_cast<uint32_t>(lvl_len_[lvl - 1]),
                    static_cast<uint32_t>(lvl_off_[lvl]), static_cast<uint32_t>(lvl_len_[lvl]),
                    hb_.center, 1.4142135623730951f};
                [enc setComputePipelineState:psoDecim_];
                [enc setBuffer:sigBuf_ offset:0 atIndex:0];
                [enc setBuffer:tapsBuf_ offset:0 atIndex:1];
                [enc setBytes:&d length:sizeof(d) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(lvl_len_[lvl], 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            const struct { uint32_t n_frames, mode; } params{static_cast<uint32_t>(n_frames_),
                                                             static_cast<uint32_t>(mode)};
            [enc setComputePipelineState:psoCqt_];
            [enc setBuffer:sigBuf_ offset:0 atIndex:0];
            [enc setBuffer:twBuf_ offset:0 atIndex:1];
            [enc setBuffer:twrBuf_ offset:0 atIndex:2];
            [enc setBuffer:soBuf_ offset:0 atIndex:3];
            [enc setBuffer:slBuf_ offset:0 atIndex:4];
            [enc setBuffer:hBuf_ offset:0 atIndex:5];
            [enc setBuffer:rbBuf_ offset:0 atIndex:6];
            [enc setBuffer:rcBuf_ offset:0 atIndex:7];
            [enc setBuffer:rsBuf_ offset:0 atIndex:8];
            [enc setBuffer:colBuf_ offset:0 atIndex:9];
            [enc setBuffer:valBuf_ offset:0 atIndex:10];
            [enc setBuffer:obBuf_ offset:0 atIndex:11];
            [enc setBuffer:outBuf_ offset:0 atIndex:12];
            [enc setBytes:&params length:sizeof(params) atIndex:13];
            [enc setBuffer:outBuf_ offset:0 atIndex:14];  // float2 view for mode 3
            [enc dispatchThreadgroups:MTLSizeMake(n_frames_, n_octaves_, 1)
                threadsPerThreadgroup:MTLSizeMake(tpb_, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error("metal: command buffer failed");
        }

        const float* result = static_cast<const float*>(outBuf_.contents);
        const std::size_t count = static_cast<std::size_t>(p_.n_bins) * n_frames_;
        if (mode == 3) {
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
            outf->data.assign(result, result + count);
        }
    }

private:
    void ensure_capacity(int n) {
        if (n == cur_n_) return;
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

        const std::size_t page = 16384;
        const std::size_t bytes =
            ((static_cast<std::size_t>(total) * sizeof(float)) + page - 1) & ~(page - 1);
        std::free(mem_);
        mem_ = nullptr;
        if (posix_memalign(&mem_, page, bytes) != 0) throw std::bad_alloc{};
        sigBuf_ = [dev_ newBufferWithBytesNoCopy:mem_
                                          length:bytes
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
        const auto mkbuf = [&](const void* ptr, std::size_t b) {
            return [dev_ newBufferWithBytes:ptr length:b options:MTLResourceStorageModeShared];
        };
        soBuf_ = mkbuf(sig_off.data(), sig_off.size() * sizeof(int));
        slBuf_ = mkbuf(sig_len.data(), sig_len.size() * sizeof(int));
        hBuf_ = mkbuf(hops.data(), hops.size() * sizeof(int));
        // Sized for the largest output (complex: two floats per element).
        outBuf_ = [dev_ newBufferWithLength:static_cast<std::size_t>(p_.n_bins) * n_frames_ * 2 * sizeof(float)
                                    options:MTLResourceStorageModeShared];
        if (!sigBuf_ || !outBuf_) throw std::runtime_error("metal: buffer allocation failed");
        cur_n_ = n;
    }

    CqtParams p_;
    int n_octaves_ = 0, n_fft_ = 0, N_ = 0, log2n_ = 0, tpb_ = 0;
    int pre_ = 0, n_levels_ = 0, hop0_ = 0;
    double base_sr_ = 0.0;
    detail::HalfBand hb_;

    id<MTLDevice> dev_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> psoCqt_, psoDecim_;
    id<MTLBuffer> twBuf_, twrBuf_, rbBuf_, rcBuf_, rsBuf_, colBuf_, valBuf_, obBuf_, tapsBuf_;

    // Sized by ensure_capacity().
    void* mem_ = nullptr;
    int cur_n_ = -1, n_frames_ = 0;
    std::vector<int> lvl_off_, lvl_len_;
    id<MTLBuffer> sigBuf_, soBuf_, slBuf_, hBuf_, outBuf_;
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

bool gpu_available() {
    static const bool ok = [] {
        @autoreleasepool {
            id<MTLDevice> dev = shared_device();
            if (!dev) return false;
            // The zero-copy pyramid design assumes unified memory (Apple
            // silicon); on discrete-GPU Macs the CPU path is the safe default.
            if (@available(macOS 10.15, *)) return static_cast<bool>(dev.hasUnifiedMemory);
            return false;
        }
    }();
    return ok;
}

const char* gpu_backend() { return "metal"; }

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

// The resident (DLPack) path is CUDA-only for now; on Apple, MPS tensors are
// staged through unified memory by the Python layer, which is nearly free.
DeviceTensor cqt_gpu_resident(DeviceTensor, int, const CqtParams&, int, std::uintptr_t, int*) {
    throw std::runtime_error("the metal backend has no device-resident path");
}
DeviceTensor chroma_gpu_resident(DeviceTensor, int, double, int, int, int, int, double,
                                 std::uintptr_t, int*) {
    throw std::runtime_error("the metal backend has no device-resident path");
}
int dlpack_device_type() { return 8; }  // kDLMetal
void device_tensor_free(void*) noexcept {}

}  // namespace auvux::fastchroma
