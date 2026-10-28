#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#ifdef USE_XPU
#include <torch/csrc/inductor/aoti_runner/model_container_runner_xpu.h>
#endif
#ifdef __APPLE__
#include <torch/csrc/inductor/aoti_runner/model_container_runner_mps.h>
#endif
#include <torch/csrc/inductor/aoti_runner/pybind.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <ATen/EmptyTensor.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace torch::inductor {

namespace {

using InductorEntryFn = void (*)(AtenTensorHandle*, AtenTensorHandle*);
using torch::aot_inductor::tensor_pointer_to_tensor_handle;

template <typename T>
at::Tensor cpu_scalar(T value, at::ScalarType dtype) {
  at::Tensor t = at::detail::empty_cpu(
      {}, dtype, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  *static_cast<T*>(t.data_ptr()) = value;
  return t;
}

// The generic wrapper's
//   arg if isinstance(arg, torch.Tensor) else torch.tensor(arg, device="cpu")
// with the common scalars converted without calling into Python.
at::Tensor to_tensor(PyObject* obj) {
  if (THPVariable_Check(obj)) {
    return THPVariable_Unpack(obj);
  }
  if (PyBool_Check(obj)) {
    return cpu_scalar<bool>(obj == Py_True, at::kBool);
  }
  if (PyLong_CheckExact(obj)) {
    int overflow = 0;
    const long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (overflow == 0 && !(value == -1 && PyErr_Occurred())) {
      return cpu_scalar<int64_t>(value, at::kLong);
    }
    PyErr_Clear();
  } else if (PyFloat_CheckExact(obj)) {
    const auto dtype = c10::typeMetaToScalarType(at::get_default_dtype());
    if (dtype == at::kFloat) {
      return cpu_scalar<float>(
          static_cast<float>(PyFloat_AS_DOUBLE(obj)), at::kFloat);
    }
    if (dtype == at::kDouble) {
      return cpu_scalar<double>(PyFloat_AS_DOUBLE(obj), at::kDouble);
    }
  }
  py::object tensor = py::module_::import("torch").attr("tensor")(
      py::handle(obj), py::arg("device") = "cpu");
  return THPVariable_Unpack(tensor.ptr());
}

// Runs a JIT cpp_wrapper graph's inductor_entry_impl on the argument list of
// its Python wrapper g(args), with the generic wrapper's semantics: args is
// cleared once every argument is converted, before the graph runs, and the
// output handles are stolen into tensors (None for null handles).
//
// The graph borrows the handles at the positions in `borrowed` (static
// inputs). For those, one handle is kept per position while the same Python
// object is passed and renewed when another object is passed; a position
// renewed on two consecutive calls gets a handle per call from then on, so a
// tensor that changes every call is not kept alive between calls. Holding
// the objects keeps their ids from being reused.
//
// Arguments are converted with the GIL held; the entry releases it. Each call
// holds the cache snapshot it converted with, so a concurrent call renewing a
// position never frees a handle in use.
class InductorEntry {
 public:
  InductorEntry(
      uintptr_t entry,
      int64_t num_outputs,
      std::vector<int64_t> borrowed)
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      : entry_(reinterpret_cast<InductorEntryFn>(entry)),
        num_outputs_(num_outputs),
        borrowed_(std::move(borrowed)),
        renewals_(borrowed_.size(), 0),
        per_call_(borrowed_.size(), false) {
    TORCH_CHECK(
        entry != 0 && num_outputs >= 0,
        "InductorEntry: invalid entry or output count");
    for (size_t b = 0; b < borrowed_.size(); ++b) {
      TORCH_CHECK(
          borrowed_[b] >= 0 && (b == 0 || borrowed_[b] > borrowed_[b - 1]),
          "InductorEntry: borrowed positions must be sorted and unique");
    }
    auto cache = std::make_shared<Cache>();
    cache->objects.resize(borrowed_.size());
    cache->tensors.resize(borrowed_.size());
    cache_ = std::move(cache);
  }

  py::list operator()(const py::list& args) {
    PyObject* list = args.ptr();
    const Py_ssize_t n = PyList_GET_SIZE(list);
    TORCH_CHECK(
        borrowed_.empty() || borrowed_.back() < n,
        "InductorEntry: ",
        n,
        " arguments, but position ",
        borrowed_.back(),
        " is borrowed");

    std::shared_ptr<const Cache> cache = cache_;
    std::shared_ptr<Cache> renewed;
    std::vector<AtenTensorHandle> inputs(n, nullptr);
    // Borrowed by the graph but owned by this call.
    std::vector<std::unique_ptr<at::Tensor>> per_call;
    // Stolen by the graph; freed here only if the call does not happen.
    OwnedHandles owned;
    owned.handles.reserve(n);
    size_t b = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
      PyObject* obj = PyList_GET_ITEM(list, i);
      if (b < borrowed_.size() && borrowed_[b] == i) {
        const Cache& current = renewed ? *renewed : *cache;
        if (current.objects[b].ptr() == obj) {
          inputs[i] = tensor_pointer_to_tensor_handle(current.tensors[b].get());
          renewals_[b] = 0;
        } else if (per_call_[b]) {
          per_call.push_back(std::make_unique<at::Tensor>(to_tensor(obj)));
          inputs[i] = tensor_pointer_to_tensor_handle(per_call.back().get());
        } else {
          at::Tensor tensor = to_tensor(obj);
          if (!renewed) {
            renewed = std::make_shared<Cache>(*cache);
          }
          const bool replaces = renewed->objects[b].ptr() != nullptr;
          if (replaces && renewals_[b] > 0) {
            per_call_[b] = true;
            renewed->objects[b] = py::object();
            renewed->tensors[b].reset();
            per_call.push_back(std::make_unique<at::Tensor>(std::move(tensor)));
            inputs[i] = tensor_pointer_to_tensor_handle(per_call.back().get());
          } else {
            renewals_[b] += replaces;
            renewed->objects[b] = py::reinterpret_borrow<py::object>(obj);
            renewed->tensors[b] =
                std::make_shared<at::Tensor>(std::move(tensor));
            inputs[i] =
                tensor_pointer_to_tensor_handle(renewed->tensors[b].get());
          }
        }
        ++b;
      } else {
        inputs[i] = torch::aot_inductor::new_tensor_handle(to_tensor(obj));
        owned.handles.push_back(inputs[i]);
      }
    }
    if (renewed) {
      cache = renewed;
      cache_ = std::move(renewed);
    }
    if (PyList_SetSlice(list, 0, n, nullptr) != 0) {
      throw py::error_already_set();
    }

    // The entry steals the owned handles, also when it throws, and releases
    // the GIL itself.
    owned.handles.clear();
    std::vector<AtenTensorHandle> outputs(num_outputs_, nullptr);
    entry_(inputs.data(), outputs.data());
    if (PyErr_Occurred()) {
      for (AtenTensorHandle h : outputs) {
        if (h != nullptr) {
          aoti_torch_delete_tensor_object(h);
        }
      }
      throw py::error_already_set();
    }
    return wrap_outputs(outputs);
  }

 private:
  struct Cache {
    std::vector<py::object> objects;
    // An AtenTensorHandle is an at::Tensor*.
    std::vector<std::shared_ptr<at::Tensor>> tensors;
  };

  struct OwnedHandles {
    std::vector<AtenTensorHandle> handles;
    OwnedHandles() = default;
    OwnedHandles(const OwnedHandles&) = delete;
    OwnedHandles& operator=(const OwnedHandles&) = delete;
    OwnedHandles(OwnedHandles&&) = delete;
    OwnedHandles& operator=(OwnedHandles&&) = delete;
    ~OwnedHandles() {
      for (AtenTensorHandle h : handles) {
        aoti_torch_delete_tensor_object(h);
      }
    }
  };

  // alloc_tensors_by_stealing_from_handles: a handle may be returned more
  // than once; it is freed at its last position.
  static py::list wrap_outputs(std::vector<AtenTensorHandle>& outputs) {
    const auto size = static_cast<int64_t>(outputs.size());
    std::unordered_map<AtenTensorHandle, int64_t> last;
    for (int64_t i = 0; i < size; ++i) {
      if (outputs[i] != nullptr) {
        last[outputs[i]] = i;
      }
    }
    py::list result(size);
    for (int64_t i = 0; i < size; ++i) {
      AtenTensorHandle h = outputs[i];
      if (h == nullptr) {
        result[i] = py::none();
        continue;
      }
      at::Tensor t = *torch::aot_inductor::tensor_handle_to_tensor_pointer(h);
      if (last[h] == i) {
        aoti_torch_delete_tensor_object(h);
      }
      result[i] =
          py::reinterpret_steal<py::object>(THPVariable_Wrap(std::move(t)));
    }
    return result;
  }

  InductorEntryFn entry_;
  int64_t num_outputs_;
  std::vector<int64_t> borrowed_;
  std::vector<uint8_t> renewals_;
  std::vector<bool> per_call_;
  std::shared_ptr<const Cache> cache_;
};

} // namespace

void initAOTIRunnerBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto m = rootModule.def_submodule("_aoti");

  py::class_<AOTIModelContainerRunnerCpu>(m, "AOTIModelContainerRunnerCpu")
      .def(py::init<const std::string&, int>())
      .def(
          "run",
          &AOTIModelContainerRunnerCpu::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelContainerRunnerCpu::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerCpu::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerCpu::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerCpu::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerCpu::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerCpu::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerCpu::swap_constant_buffer)
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerCpu::free_inactive_constant_buffer)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerCpu::update_constant_buffer_from_blob,
          py::arg("weights_path"));

#ifdef USE_CUDA
  py::class_<AOTIModelContainerRunnerCuda>(m, "AOTIModelContainerRunnerCuda")
      .def(py::init<const std::string&, int>())
      .def(py::init<const std::string&, int, const std::string&>())
      .def(py::init<
           const std::string&,
           int,
           const std::string&,
           const std::string&>())
      .def(py::init<
           const std::string&,
           int,
           const std::string&,
           const std::string&,
           const bool>())
      .def(
          "run",
          &AOTIModelContainerRunnerCuda::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelContainerRunnerCuda::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerCuda::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerCuda::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerCuda::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerCuda::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerCuda::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "update_constant_buffer_from_cpu",
          static_cast<void (AOTIModelContainerRunnerCuda::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool)>(
              &AOTIModelContainerRunnerCuda::update_constant_buffer_from_cpu),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"))
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerCuda::swap_constant_buffer)
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerCuda::free_inactive_constant_buffer)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerCuda::update_constant_buffer_from_blob,
          py::arg("weights_path"));
#endif
#ifdef USE_XPU
  py::class_<AOTIModelContainerRunnerXpu>(m, "AOTIModelContainerRunnerXpu")
      .def(py::init<const std::string&, int>())
      .def(py::init<const std::string&, int, const std::string&>())
      .def(py::init<
           const std::string&,
           int,
           const std::string&,
           const std::string&>())
      .def(
          "run",
          &AOTIModelContainerRunnerXpu::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelContainerRunnerXpu::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerXpu::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerXpu::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerXpu::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerXpu::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerXpu::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "update_constant_buffer_from_cpu",
          static_cast<void (AOTIModelContainerRunnerXpu::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool)>(
              &AOTIModelContainerRunnerXpu::update_constant_buffer_from_cpu),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"))
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerXpu::swap_constant_buffer)
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerXpu::free_inactive_constant_buffer)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerXpu::update_constant_buffer_from_blob,
          py::arg("weights_path"));
#endif
#if defined(USE_MPS) && defined(__APPLE__) && \
    !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
  py::class_<AOTIModelContainerRunnerMps>(m, "AOTIModelContainerRunnerMps")
      .def(py::init<const std::string&, int>())
      .def(
          "run",
          &AOTIModelContainerRunnerMps::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelContainerRunnerMps::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerMps::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerMps::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerMps::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerMps::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerMps::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "update_constant_buffer_from_cpu",
          static_cast<void (AOTIModelContainerRunnerMps::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool)>(
              &AOTIModelContainerRunnerMps::update_constant_buffer_from_cpu),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"))
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerMps::swap_constant_buffer)
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerMps::free_inactive_constant_buffer)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerMps::update_constant_buffer_from_blob,
          py::arg("weights_path"));
#endif

  py::class_<InductorEntry>(m, "InductorEntry")
      .def(
          py::init<uintptr_t, int64_t, std::vector<int64_t>>(),
          py::arg("entry"),
          py::arg("num_outputs"),
          py::arg("borrowed") = std::vector<int64_t>{})
      .def("__call__", &InductorEntry::operator(), py::arg("args"));

  m.def(
      "unsafe_alloc_void_ptrs_from_tensors",
      [](const std::vector<at::Tensor>& tensors) {
        std::vector<AtenTensorHandle> handles =
            torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(tensors);
        std::vector<void*> result(
            reinterpret_cast<void**>(handles.data()),
            reinterpret_cast<void**>(handles.data()) + handles.size());
        return result;
      });
  m.def("unsafe_alloc_void_ptr_from_tensor", [](at::Tensor& tensor) {
    return reinterpret_cast<void*>(
        torch::aot_inductor::new_tensor_handle(std::move(tensor)));
  });
  m.def(
      "alloc_tensors_by_stealing_from_void_ptrs",
      [](std::vector<void*>& raw_handles) {
        return torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
            reinterpret_cast<AtenTensorHandle*>(raw_handles.data()),
            raw_handles.size());
      });
  m.def("alloc_tensor_by_stealing_from_void_ptr", [](void* raw_handle) {
    return *torch::aot_inductor::tensor_handle_to_tensor_pointer(
        reinterpret_cast<AtenTensorHandle>(raw_handle));
  });
}
} // namespace torch::inductor
