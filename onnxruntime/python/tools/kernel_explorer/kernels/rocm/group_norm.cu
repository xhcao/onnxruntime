// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "contrib_ops/rocm/diffusion/group_norm_ck.cuh"
#include "contrib_ops/rocm/diffusion/group_norm_common.h"
#include "contrib_ops/rocm/diffusion/group_norm_tunable_op.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;
using onnxruntime::contrib::rocm::GetGroupNormWorkspaceSizeInBytes;
namespace onnxruntime {

template <typename T, int ThreadsPerBlock, int VecSize>
class GroupNormNHWC : public IKernelExplorer {
 public:
  GroupNormNHWC(DeviceArray& output, DeviceArray& add_output, DeviceArray& input, DeviceArray& skip, DeviceArray& bias,
                DeviceArray& gamma, DeviceArray& beta, DeviceArray& workspace, float epsilon,
                int batch_size, int num_channels, int height, int width, int num_groups, bool use_silu,
                bool broadcast_skip, int channels_per_block)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(add_output.ptr()),
                static_cast<T*>(input.ptr()), static_cast<T*>(skip.ptr()), static_cast<T*>(bias.ptr()),
                static_cast<float*>(gamma.ptr()), static_cast<float*>(beta.ptr()), static_cast<float*>(workspace.ptr()),
                epsilon, batch_size, num_channels, height, width, num_groups, use_silu, broadcast_skip,
                channels_per_block) {
    type_string_ = "GroupNormNHWC_" + std::to_string(ThreadsPerBlock) + "_" + std::to_string(VecSize);
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  std::vector<std::string> ListOps() const {
    return {type_string_};
  }

  bool SelectOp(const std::string& name) {
    Status status = op_.IsSupported(&params_);
    return status.IsOK() && name == type_string_;
  }

 private:
  using ParamsT = contrib::rocm::GroupNormNHWCTunableParams<T>;
  ParamsT params_{};
  contrib::rocm::GroupNormNHWCOp<T, ThreadsPerBlock, VecSize> op_{};
  std::string type_string_{};
};

template <typename T>
class GroupNormNHWCStaticSelection : public IKernelExplorer {
 public:
  GroupNormNHWCStaticSelection(DeviceArray& output, DeviceArray& add_output, DeviceArray& input, DeviceArray& skip,
                               DeviceArray& bias, DeviceArray& gamma, DeviceArray& beta, DeviceArray& workspace,
                               float epsilon, int batch_size, int num_channels, int height, int width, int num_groups,
                               bool use_silu, bool broadcast_skip, int channels_per_block)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(add_output.ptr()),
                static_cast<T*>(input.ptr()), static_cast<T*>(skip.ptr()), static_cast<T*>(bias.ptr()),
                static_cast<float*>(gamma.ptr()), static_cast<float*>(beta.ptr()), static_cast<float*>(workspace.ptr()),
                epsilon, batch_size, num_channels, height, width, num_groups, use_silu, broadcast_skip,
                channels_per_block) {
    type_string_ = "GroupNormNHWCStaticSelection";
  }

  void Run() override {
    ORT_THROW_IF_ERROR((contrib::rocm::GroupNormNHWCStaticSelection<T>(&params_)));
  }

  std::vector<std::string> ListOps() const {
    return {type_string_};
  }

  bool SelectOp(const std::string& name) {
    Status status = contrib::rocm::GroupNormNHWCStaticSelection<T>(&params_);
    return status.IsOK() && name == type_string_;
  }

 private:
  using ParamsT = contrib::rocm::GroupNormNHWCTunableParams<T>;
  ParamsT params_{};
  std::string type_string_{};
};

template <typename T>
class GroupNormNHWCTunable : public IKernelExplorer {
 public:
  GroupNormNHWCTunable(DeviceArray& output, DeviceArray& add_output, DeviceArray& input, DeviceArray& skip,
                       DeviceArray& bias, DeviceArray& gamma, DeviceArray& beta, DeviceArray& workspace,
                       float epsilon, int batch_size, int num_channels, int height, int width, int num_groups,
                       bool use_silu, bool broadcast_skip, int channels_per_block)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(add_output.ptr()),
                static_cast<T*>(input.ptr()), static_cast<T*>(skip.ptr()), static_cast<T*>(bias.ptr()),
                static_cast<float*>(gamma.ptr()), static_cast<float*>(beta.ptr()), static_cast<float*>(workspace.ptr()),
                epsilon, batch_size, num_channels, height, width, num_groups, use_silu, broadcast_skip,
                channels_per_block) {
    params_.TuningContext()->EnableTunableOpAndTuning();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  std::vector<std::string> ListOps() const {
    return {"GroupNormNHWCTunable"};
  }

  bool SelectOp(const std::string& name) {
    return name == "GroupNormNHWCTunable";
  }

 private:
  using ParamsT = contrib::rocm::GroupNormNHWCTunableParams<T>;
  ParamsT params_{};
  contrib::rocm::GroupNormNHWCTunableOp<T> op_{};
};

#ifdef USE_COMPOSABLE_KERNEL
template <typename T, bool WithSilu>
class CKGroupNormNHWC : public IKernelExplorer {
 public:
  CKGroupNormNHWC(DeviceArray& output, DeviceArray& add_output, DeviceArray& input, DeviceArray& skip,
                  DeviceArray& bias, DeviceArray& gamma, DeviceArray& beta, DeviceArray& workspace,
                  float epsilon, int batch_size, int num_channels, int height, int width, int num_groups,
                  bool use_silu, bool broadcast_skip, int channels_per_block)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(add_output.ptr()),
                static_cast<T*>(input.ptr()), static_cast<T*>(skip.ptr()), static_cast<T*>(bias.ptr()),
                static_cast<float*>(gamma.ptr()), static_cast<float*>(beta.ptr()), static_cast<float*>(workspace.ptr()),
                epsilon, batch_size, num_channels, height, width, num_groups, use_silu, broadcast_skip,
                channels_per_block) {
    for (auto&& [type_string, op] : contrib::rocm::GetCKGroupNormNHWCTypeStringAndOps<T, float, WithSilu>()) {
      type_strings_.emplace_back(std::move(type_string));
      ops_.emplace_back(std::move(op));
    }
  }

  void Run() override {
    ORT_THROW_IF_ERROR(ops_[selected_op_](&params_));
  }

  std::vector<std::string> ListOps() const {
    return type_strings_;
  }

  bool SelectOp(const std::string& name) {
    for (size_t i = 0; i < ops_.size(); i++) {
      if (type_strings_[i] == name) {
        selected_op_ = i;
        Status status = ops_[i](&params_);
        return status.IsOK();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

 private:
  using ParamsT = contrib::rocm::GroupNormNHWCTunableParams<T>;
  using OpT = rocm::tunable::Op<ParamsT>;
  ParamsT params_{};
  std::vector<OpT> ops_;
  std::vector<std::string> type_strings_;
  size_t selected_op_{};
};
#endif  // USE_COMPOSABLE_KERNEL

#ifdef USE_TRITON_KERNEL
template <typename T, bool WithSilu>
class GroupNormNHWCTriton : public IKernelExplorer {
 public:
  GroupNormNHWCTriton(DeviceArray& output, DeviceArray& add_output, DeviceArray& input, DeviceArray& skip,
                      DeviceArray& bias, DeviceArray& gamma, DeviceArray& beta, DeviceArray& workspace,
                      float epsilon, int batch_size, int num_channels, int height, int width, int num_groups,
                      bool use_silu, bool broadcast_skip, int channels_per_block)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(add_output.ptr()),
                static_cast<T*>(input.ptr()), static_cast<T*>(skip.ptr()), static_cast<T*>(bias.ptr()),
                static_cast<float*>(gamma.ptr()), static_cast<float*>(beta.ptr()), static_cast<float*>(workspace.ptr()),
                epsilon, batch_size, num_channels, height, width, num_groups, use_silu, broadcast_skip,
                channels_per_block) {
    for (auto&& [name, op] : contrib::rocm::GetTritonGroupNormNHWCTypeStringAndOps<T, WithSilu>()) {
      name_strings_.emplace_back(name);
      ops_.emplace_back(std::move(op));
    }
  }

  void Run() override {
    ORT_THROW_IF_ERROR(ops_[selected_op_](&params_));
  }

  std::vector<std::string> ListOps() const {
    return name_strings_;
  }

  bool SelectOp(const std::string& name) {
    for (size_t i = 0; i < ops_.size(); i++) {
      if (name_strings_[i] == name) {
        selected_op_ = i;
        Status status = ops_[i](&params_);
        return status.IsOK();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

 private:
  using ParamsT = contrib::rocm::GroupNormNHWCTunableParams<T>;
  using OpT = rocm::tunable::Op<ParamsT>;
  ParamsT params_{};
  std::vector<OpT> ops_;
  std::vector<std::string> name_strings_;
  size_t selected_op_{};
};
#endif  // USE_TRITON_KERNEL

#define REGISTER_OP(name, type, threads_per_block, vec_size)                                                   \
  py::class_<name<type, threads_per_block, vec_size>>(m, #name "_" #type "_" #threads_per_block "_" #vec_size) \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&,                      \
                    DeviceArray&, DeviceArray&, DeviceArray&, float,                                           \
                    int, int, int, int, int, bool, bool, int>())                                               \
      .def("SetRepeats", &name<type, threads_per_block, vec_size>::SetRepeats)                                 \
      .def("Profile", &name<type, threads_per_block, vec_size>::Profile)                                       \
      .def("Run", &name<type, threads_per_block, vec_size>::Run)                                               \
      .def("ListOps", &name<type, threads_per_block, vec_size>::ListOps)                                       \
      .def("SelectOp", &name<type, threads_per_block, vec_size>::SelectOp);

#define REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, threads_per_block) \
  REGISTER_OP(name, type, threads_per_block, 1)                     \
  REGISTER_OP(name, type, threads_per_block, 2)                     \
  REGISTER_OP(name, type, threads_per_block, 4)

#define REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name, type) \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 64)                         \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 128)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 192)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 256)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 320)

#define REGISTER_COMMON(name, type, ...)                                                  \
  py::class_<type<__VA_ARGS__>>(m, name)                                                  \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, \
                    DeviceArray&, DeviceArray&, DeviceArray&, float,                      \
                    int, int, int, int, int, bool, bool, int>())                          \
      .def("SetRepeats", &type<__VA_ARGS__>::SetRepeats)                                  \
      .def("Profile", &type<__VA_ARGS__>::Profile)                                        \
      .def("Run", &type<__VA_ARGS__>::Run)                                                \
      .def("ListOps", &type<__VA_ARGS__>::ListOps)                                        \
      .def("SelectOp", &type<__VA_ARGS__>::SelectOp);

#define REGISTER_OP_TYPED(name, type) \
  REGISTER_COMMON(#name "_" #type, name, type)

#define REGISTER_CK(type, with_silu, silu_suffix) \
  REGISTER_COMMON("CKGroupNormNHWC" silu_suffix "_" #type, CKGroupNormNHWC, type, with_silu)

#define REGISTER_TRITON(type, with_silu, silu_suffix) \
  REGISTER_COMMON("GroupNormNHWCTriton" silu_suffix "_" #type, GroupNormNHWCTriton, type, with_silu)

KE_REGISTER(m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(GroupNormNHWC, half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(GroupNormNHWC, float);

  REGISTER_OP_TYPED(GroupNormNHWCTunable, half);
  REGISTER_OP_TYPED(GroupNormNHWCTunable, float);

  REGISTER_OP_TYPED(GroupNormNHWCStaticSelection, half);
  REGISTER_OP_TYPED(GroupNormNHWCStaticSelection, float);

#ifdef USE_COMPOSABLE_KERNEL
  REGISTER_CK(half, false, "Pass");
  REGISTER_CK(half, true, "Silu");
  REGISTER_CK(float, false, "Pass");
  REGISTER_CK(float, true, "Silu");
#endif  // USE_COMPOSABLE_KERNEL

#ifdef USE_TRITON_KERNEL
  REGISTER_TRITON(half, false, "Pass");
  REGISTER_TRITON(half, true, "Silu");
  REGISTER_TRITON(float, false, "Pass");
  REGISTER_TRITON(float, true, "Silu");
#endif
}

}  // namespace onnxruntime
