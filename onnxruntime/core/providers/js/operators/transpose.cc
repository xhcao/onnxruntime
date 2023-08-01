// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transpose.h"

namespace onnxruntime {
namespace js {

#define REG_ELEMENTWISE_TYPED_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS)         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                   \
      OP_TYPE,                                                                     \
      kOnnxDomain,                                                                 \
      VERSION,                                                                     \
      TYPE,                                                                        \
      kJsExecutionProvider,                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      KERNEL_CLASS);

#define REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                            \
      OP_TYPE,                                                                                        \
      kOnnxDomain,                                                                                    \
      VERSION_FROM, VERSION_TO,                                                                       \
      TYPE,                                                                                           \
      kJsExecutionProvider,                                                                           \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()),                    \
      KERNEL_CLASS);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Transpose, 1, 12, float, Transpose);
REG_ELEMENTWISE_TYPED_KERNEL(Transpose, 13, float, Transpose);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Transpose, 1, 12, int32_t, Transpose);
REG_ELEMENTWISE_TYPED_KERNEL(Transpose, 13, int32_t, Transpose);

}  // namespace js
}  // namespace onnxruntime
