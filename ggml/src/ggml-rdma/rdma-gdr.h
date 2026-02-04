#pragma once

#include "rdma-transport.h"
#include "rdma-memory.h"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace ggml_rdma {

// GPU memory region information
struct gpu_memory_region {
    void *          gpu_addr;       // GPU virtual address
    void *          host_addr;      // Mapped host address (if applicable)
    size_t          size;           // Size of the region
    struct ibv_mr * mr;             // RDMA memory region
    int             device_id;      // CUDA device ID
    bool            peer_mapped;    // Whether nvidia-peermem mapping is active
};

// GPUDirect RDMA memory manager
// Handles GPU memory registration for RDMA access via nvidia-peermem
class gdr_memory_manager {
public:
    gdr_memory_manager(rdma_connection * conn);
    ~gdr_memory_manager();

    // Disable copy
    gdr_memory_manager(const gdr_memory_manager&) = delete;
    gdr_memory_manager& operator=(const gdr_memory_manager&) = delete;

    // Check if GPUDirect RDMA is available on this system
    static bool is_available();

    // Check if nvidia-peermem module is loaded
    static bool is_peermem_loaded();

    // Initialize GPUDirect for a specific GPU device
    bool init_device(int device_id);

    // Register GPU memory for RDMA access
    // The GPU memory must already be allocated (e.g., via cudaMalloc)
    // Returns nullptr on failure
    gpu_memory_region * register_gpu_memory(void * gpu_addr, size_t size, int device_id);

    // Deregister GPU memory
    void deregister_gpu_memory(gpu_memory_region * region);

    // Allocate GPU memory and register for RDMA
    gpu_memory_region * alloc_gpu_memory(size_t size, int device_id);

    // Free GPU memory allocated by alloc_gpu_memory
    void free_gpu_memory(gpu_memory_region * region);

    // Get memory region by GPU address
    gpu_memory_region * get_region(void * gpu_addr);

    // Get remote memory info for RDMA operations
    remote_memory_info get_remote_info(gpu_memory_region * region);

    // Clear all registrations
    void clear();

    // Check if a device has been initialized
    bool is_device_initialized(int device_id) const;

private:
    rdma_connection * conn_;
    std::mutex mutex_;

    // Registered GPU memory regions
    std::vector<std::unique_ptr<gpu_memory_region>> regions_;

    // Map GPU address to region for fast lookup
    std::unordered_map<void *, gpu_memory_region *> addr_map_;

    // Initialized device IDs
    std::vector<int> initialized_devices_;

    // Check nvidia-peermem support
    static bool check_peermem_support();
};

// Pinned host memory manager for CPU-GPU data staging
// This is used when GPUDirect is not available or for control data
class pinned_memory_manager {
public:
    pinned_memory_manager();
    ~pinned_memory_manager();

    // Disable copy
    pinned_memory_manager(const pinned_memory_manager&) = delete;
    pinned_memory_manager& operator=(const pinned_memory_manager&) = delete;

    // Allocate pinned host memory
    void * alloc(size_t size);

    // Free pinned host memory
    void free(void * ptr);

    // Get or allocate staging buffer of at least 'size' bytes
    void * get_staging_buffer(size_t size);

private:
    std::mutex mutex_;
    std::vector<std::pair<void *, size_t>> allocations_;

    // Reusable staging buffer
    void * staging_buffer_;
    size_t staging_buffer_size_;
};

} // namespace ggml_rdma
