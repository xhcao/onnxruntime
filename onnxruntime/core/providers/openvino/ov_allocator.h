// Copyright (C) Intel Corporation
// Licensed under the MIT License
#ifdef USE_OVEP_NPU_MEMORY
#pragma once

#include <mutex>

#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include "openvino/runtime/remote_context.hpp"
#include "core/common/inlined_containers.h"

namespace onnxruntime {

class OVRTAllocator : public IAllocator {
 public:
  OVRTAllocator(ov::Core& core, OrtDevice::DeviceType device_type, OrtDevice::DeviceId device_id, const char* name);
  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  ov::Core& core_;
  ov::RemoteContext remote_ctx_;
  InlinedHashMap<void*, ov::Tensor*> allocated_;
  std::mutex mutex_;
};

}  // namespace onnxruntime
#endif
