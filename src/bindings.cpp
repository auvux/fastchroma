#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <climits>
#include <complex>
#include <cstring>

#include "fastchroma.hpp"
#include "third_party/dlpack.h"

namespace py = pybind11;
using auvux::fastchroma::ComplexMatrix;
using auvux::fastchroma::DeviceTensor;
using auvux::fastchroma::Matrix;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

py::array_t<float> to_numpy(const Matrix& m) {
    py::array_t<float> out({m.rows, m.cols});
    std::copy(m.data.begin(), m.data.end(), out.mutable_data());
    return out;
}

py::array_t<float> chroma(FloatArray y, double sr, int hop, int bins_per_octave,
                              int n_octaves, int n_chroma, double fmin) {
    const auto buf = y.request();
    const auto* data = static_cast<const float*>(buf.ptr);
    const auto n = static_cast<std::size_t>(buf.size);
    Matrix m;
    {
        const py::gil_scoped_release release;
        m = auvux::fastchroma::chroma(data, n, sr, hop, bins_per_octave, n_octaves, n_chroma, fmin);
    }
    return to_numpy(m);
}

auvux::fastchroma::CqtParams make_params(double sr, int hop, double fmin, int n_bins,
                                         int bins_per_octave) {
    auvux::fastchroma::CqtParams params;
    params.sr = sr;
    params.hop = hop;
    params.fmin = fmin;
    params.n_bins = n_bins;
    params.bins_per_octave = bins_per_octave;
    return params;
}

py::array_t<float> cqt(FloatArray y, double sr, int hop, double fmin, int n_bins,
                                 int bins_per_octave) {
    const auto buf = y.request();
    const auto* data = static_cast<const float*>(buf.ptr);
    const auto n = static_cast<std::size_t>(buf.size);
    const auto params = make_params(sr, hop, fmin, n_bins, bins_per_octave);
    Matrix m;
    {
        const py::gil_scoped_release release;
        m = auvux::fastchroma::cqt(data, n, params);
    }
    return to_numpy(m);
}

// GPU variants return None when the backend is unavailable or the parameters
// are unsupported; the Python layer decides whether to fall back or raise.
py::object cqt_gpu(FloatArray y, double sr, int hop, double fmin, int n_bins,
                   int bins_per_octave, int output_mode) {
    const auto buf = y.request();
    const auto* data = static_cast<const float*>(buf.ptr);
    const auto n = static_cast<std::size_t>(buf.size);
    const auto params = make_params(sr, hop, fmin, n_bins, bins_per_octave);
    Matrix m;
    bool ok = false;
    {
        const py::gil_scoped_release release;
        ok = auvux::fastchroma::cqt_gpu(data, n, params, m, output_mode);
    }
    if (!ok) return py::none();
    return to_numpy(m);
}

py::object cqt_complex_gpu(FloatArray y, double sr, int hop, double fmin, int n_bins,
                           int bins_per_octave) {
    const auto buf = y.request();
    const auto* data = static_cast<const float*>(buf.ptr);
    const auto n = static_cast<std::size_t>(buf.size);
    const auto params = make_params(sr, hop, fmin, n_bins, bins_per_octave);
    ComplexMatrix m;
    bool ok = false;
    {
        const py::gil_scoped_release release;
        ok = auvux::fastchroma::cqt_complex_gpu(data, n, params, m);
    }
    if (!ok) return py::none();
    py::array_t<std::complex<float>> out({m.rows, m.cols});
    auto* o = out.mutable_data();
    for (std::size_t i = 0; i < m.re.size(); ++i) o[i] = {m.re[i], m.im[i]};
    return out;
}

py::object chroma_gpu(FloatArray y, double sr, int hop, int bins_per_octave,
                          int n_octaves, int n_chroma, double fmin) {
    const auto buf = y.request();
    const auto* data = static_cast<const float*>(buf.ptr);
    const auto n = static_cast<std::size_t>(buf.size);
    Matrix m;
    bool ok = false;
    {
        const py::gil_scoped_release release;
        ok = auvux::fastchroma::chroma_gpu(data, n, sr, hop, bins_per_octave, n_octaves,
                                               n_chroma, fmin, m);
    }
    if (!ok) return py::none();
    return to_numpy(m);
}

py::array_t<std::complex<float>> cqt_complex(FloatArray y, double sr, int hop, double fmin,
                                             int n_bins, int bins_per_octave) {
    const auto buf = y.request();
    const auto* data = static_cast<const float*>(buf.ptr);
    const auto n = static_cast<std::size_t>(buf.size);
    const auto params = make_params(sr, hop, fmin, n_bins, bins_per_octave);
    ComplexMatrix m;
    {
        const py::gil_scoped_release release;
        m = auvux::fastchroma::cqt_complex(data, n, params);
    }
    py::array_t<std::complex<float>> out({m.rows, m.cols});
    auto* o = out.mutable_data();
    for (std::size_t i = 0; i < m.re.size(); ++i) o[i] = {m.re[i], m.im[i]};
    return out;
}

// ---- GPU-resident path (DLPack), mirroring fastmel ----
//
// Consumes a DLPack "dltensor" capsule (the device pointer stays on the GPU)
// and returns a new "dltensor" capsule of the result on the same device. The
// Python wrapper handles framework conversion and stream synchronization.

struct DLInput {
    DeviceTensor t;
    int n;
};

DLInput dlpack_input(const py::capsule& input) {
    if (std::strcmp(input.name(), "dltensor") != 0)
        throw std::invalid_argument("expected a DLPack 'dltensor' capsule");
    auto* in = static_cast<DLManagedTensor*>(input.get_pointer());
    const DLTensor& t = in->dl_tensor;
    if (t.ndim != 1) throw std::invalid_argument("y must be 1-D");
    if (t.dtype.code != kDLFloat || t.dtype.bits != 32 || t.dtype.lanes != 1)
        throw std::invalid_argument("y must be float32");
    if (t.strides && t.strides[0] != 1)
        throw std::invalid_argument("y must be contiguous");
    if (static_cast<int>(t.device.device_type) != auvux::fastchroma::dlpack_device_type())
        throw std::invalid_argument(
            "tensor device does not match this build's GPU backend");
    if (t.shape[0] > INT_MAX) throw std::invalid_argument("y too long");
    return {{t.data, static_cast<std::size_t>(t.byte_offset)},
            static_cast<int>(t.shape[0])};
}

// Take ownership per the DLPack protocol. With an external stream the kernel
// may still be pending, but releasing at enqueue time matches native
// framework ops: the producer's allocator reuses freed blocks stream-ordered,
// which is exactly what orders them after our kernel.
void consume_dlpack_input(const py::capsule& input) {
    auto* in = static_cast<DLManagedTensor*>(input.get_pointer());
    PyCapsule_SetName(input.ptr(), "used_dltensor");
    if (in->deleter) in->deleter(in);
}

// DLManagedTensor for the output, with the shape/stride storage inline so
// one allocation covers everything the deleter must free.
struct DLPackOut {
    DLManagedTensor mt{};
    int64_t shape[2];
    int64_t strides[2];
};

py::capsule dlpack_output(DeviceTensor result, int rows, int n_frames, bool complex_out) {
    auto* holder = new DLPackOut();
    holder->shape[0] = rows;
    holder->shape[1] = n_frames;
    holder->strides[0] = n_frames;  // DLPack strides are in elements
    holder->strides[1] = 1;
    holder->mt.dl_tensor.data = result.handle;
    holder->mt.dl_tensor.device = {
        static_cast<DLDeviceType>(auvux::fastchroma::dlpack_device_type()), 0};
    holder->mt.dl_tensor.ndim = 2;
    holder->mt.dl_tensor.dtype =
        complex_out ? DLDataType{kDLComplex, 64, 1} : DLDataType{kDLFloat, 32, 1};
    holder->mt.dl_tensor.shape = holder->shape;
    holder->mt.dl_tensor.strides = holder->strides;
    holder->mt.dl_tensor.byte_offset = result.byte_offset;
    holder->mt.manager_ctx = holder;
    holder->mt.deleter = [](DLManagedTensor* self) {
        auto* h = static_cast<DLPackOut*>(self->manager_ctx);
        auvux::fastchroma::device_tensor_free(h->mt.dl_tensor.data);
        delete h;
    };
    // If the capsule is never consumed (renamed to "used_dltensor"), its
    // destructor must release the tensor.
    return py::capsule(&holder->mt, "dltensor", [](PyObject* cap) {
        if (PyCapsule_IsValid(cap, "dltensor")) {
            auto* mt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
            if (mt && mt->deleter) mt->deleter(mt);
        } else {
            PyErr_Clear();
        }
    });
}

py::capsule cqt_dlpack(const py::capsule& input, double sr, int hop, double fmin,
                       int n_bins, int bins_per_octave, int output_mode,
                       std::uintptr_t stream) {
    const auto params = make_params(sr, hop, fmin, n_bins, bins_per_octave);
    const DLInput in = dlpack_input(input);
    DeviceTensor result;
    int n_frames = 0;
    {
        const py::gil_scoped_release release;
        result = auvux::fastchroma::cqt_gpu_resident(in.t, in.n, params, output_mode,
                                                     stream, &n_frames);
    }
    consume_dlpack_input(input);
    return dlpack_output(result, n_bins, n_frames, output_mode == 3);
}

py::capsule chroma_dlpack(const py::capsule& input, double sr, int hop,
                          int bins_per_octave, int n_octaves, int n_chroma, double fmin,
                          std::uintptr_t stream) {
    const DLInput in = dlpack_input(input);
    DeviceTensor result;
    int n_frames = 0;
    {
        const py::gil_scoped_release release;
        result = auvux::fastchroma::chroma_gpu_resident(
            in.t, in.n, sr, hop, bins_per_octave, n_octaves, n_chroma, fmin, stream,
            &n_frames);
    }
    consume_dlpack_input(input);
    return dlpack_output(result, n_chroma, n_frames, false);
}

}  // namespace

PYBIND11_MODULE(_fastchroma, m) {
    m.def("chroma", &chroma, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("bins_per_octave"), py::arg("n_octaves"), py::arg("n_chroma"), py::arg("fmin"));
    m.def("cqt", &cqt, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("fmin"), py::arg("n_bins"), py::arg("bins_per_octave"));
    m.def("cqt_complex", &cqt_complex, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("fmin"), py::arg("n_bins"), py::arg("bins_per_octave"));
    m.def("gpu_available", &auvux::fastchroma::gpu_available);
    m.def("gpu_backend", &auvux::fastchroma::gpu_backend);
    m.def("cqt_gpu", &cqt_gpu, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("fmin"), py::arg("n_bins"), py::arg("bins_per_octave"),
          py::arg("output_mode") = 0);
    m.def("cqt_complex_gpu", &cqt_complex_gpu, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("fmin"), py::arg("n_bins"), py::arg("bins_per_octave"));
    m.def("chroma_gpu", &chroma_gpu, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("bins_per_octave"), py::arg("n_octaves"), py::arg("n_chroma"), py::arg("fmin"));
    m.def("dlpack_device_type", &auvux::fastchroma::dlpack_device_type);
    m.def("cqt_dlpack", &cqt_dlpack, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("fmin"), py::arg("n_bins"), py::arg("bins_per_octave"),
          py::arg("output_mode") = 0, py::arg("stream") = 0);
    m.def("chroma_dlpack", &chroma_dlpack, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("bins_per_octave"), py::arg("n_octaves"), py::arg("n_chroma"),
          py::arg("fmin"), py::arg("stream") = 0);
}
