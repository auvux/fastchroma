#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>

#include "fastchroma.hpp"

namespace py = pybind11;
using auvux::fastchroma::ComplexMatrix;
using auvux::fastchroma::Matrix;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

py::array_t<float> to_numpy(const Matrix& m) {
    py::array_t<float> out({m.rows, m.cols});
    std::copy(m.data.begin(), m.data.end(), out.mutable_data());
    return out;
}

py::array_t<float> chroma_cqt(FloatArray y, double sr, int hop, int bins_per_octave,
                              int n_octaves, int n_chroma, double fmin) {
    const auto buf = y.request();
    const auto* data = static_cast<const float*>(buf.ptr);
    const auto n = static_cast<std::size_t>(buf.size);
    Matrix m;
    {
        const py::gil_scoped_release release;
        m = auvux::fastchroma::chroma_cqt(data, n, sr, hop, bins_per_octave, n_octaves, n_chroma, fmin);
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

py::array_t<float> cqt_magnitude(FloatArray y, double sr, int hop, double fmin, int n_bins,
                                 int bins_per_octave) {
    const auto buf = y.request();
    const auto* data = static_cast<const float*>(buf.ptr);
    const auto n = static_cast<std::size_t>(buf.size);
    const auto params = make_params(sr, hop, fmin, n_bins, bins_per_octave);
    Matrix m;
    {
        const py::gil_scoped_release release;
        m = auvux::fastchroma::cqt_magnitude(data, n, params);
    }
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

}  // namespace

PYBIND11_MODULE(_fastchroma, m) {
    m.def("chroma_cqt", &chroma_cqt, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("bins_per_octave"), py::arg("n_octaves"), py::arg("n_chroma"), py::arg("fmin"));
    m.def("cqt_magnitude", &cqt_magnitude, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("fmin"), py::arg("n_bins"), py::arg("bins_per_octave"));
    m.def("cqt_complex", &cqt_complex, py::arg("y"), py::arg("sr"), py::arg("hop"),
          py::arg("fmin"), py::arg("n_bins"), py::arg("bins_per_octave"));
}
