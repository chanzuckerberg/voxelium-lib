#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "torch_extensions/inplace_topk/inplace_topk.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "inplace_topk",
        &inplace_topk,
        "Inplace TopK",
        py::arg("top_values"),
        py::arg("top_indices"),
        py::arg("min_top_indices"),
        py::arg("sums"),
        py::arg("squared_sums"),
        py::arg("candidate_values"),
        py::arg("candidate_indices")
    );
}
