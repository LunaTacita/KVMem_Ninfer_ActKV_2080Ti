#pragma once
#include "core/device.h"
#include "ops/common/memory.cuh"

namespace ninfer::ops::detail {
template <int Bytes>
__device__ __forceinline__ unsigned char* q8_shared_storage() {
    static_assert(Bytes <= kMaxDynamicSharedMemoryBytes);
    if constexpr (Bytes > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char dynamic_shared[];
        return dynamic_shared;
    } else {
        __shared__ __align__(16) unsigned char storage[Bytes];
        return storage;
    }
}

template <int Bytes, auto Kernel>
int q8_prepare_shared() {
    if constexpr (Bytes > 48 * 1024) {
        configure_cuda_device_once([] {
            return cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
        });
        return Bytes;
    } else
        return 0;
}
} // namespace ninfer::ops::detail
