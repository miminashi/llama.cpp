#pragma once

#include "rdma-transport.h"
#include <infiniband/verbs.h>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace ggml_rdma {

// Memory allocation type
enum class alloc_type {
    HOST,           // Regular host memory
    HOST_PINNED,    // Pinned host memory (for CUDA)
    GPU_DIRECT      // GPU memory registered for RDMA
};

// Memory region information with metadata
struct memory_region_info {
    void *          addr;       // Local virtual address
    size_t          size;       // Size of the region
    struct ibv_mr * mr;         // Memory region handle
    bool            is_gpu;     // Whether this is GPU memory (GPUDirect)
    int             gpu_device; // GPU device ID if is_gpu
    uint64_t        remote_ptr; // Remote buffer pointer (for tracking)
    alloc_type      type;       // Allocation type for proper deallocation
};

// RDMA memory pool - manages memory allocations and registrations
class rdma_memory_pool {
public:
    rdma_memory_pool(rdma_connection * conn);
    ~rdma_memory_pool();

    // Disable copy
    rdma_memory_pool(const rdma_memory_pool&) = delete;
    rdma_memory_pool& operator=(const rdma_memory_pool&) = delete;

    // Allocate and register memory
    // Returns nullptr on failure
    memory_region_info * alloc(size_t size, alloc_type type = alloc_type::HOST);

    // Free previously allocated memory
    void free(memory_region_info * info);

    // Register existing memory (does not take ownership)
    // access_flags: IBV_ACCESS_* flags
    memory_region_info * register_memory(void * addr, size_t size, int access_flags);

    // Deregister memory (does not free underlying memory)
    void deregister_memory(memory_region_info * info);

    // Get memory region for a remote buffer pointer
    memory_region_info * get_region(uint64_t remote_ptr);

    // Get all registered regions (for debugging)
    std::vector<memory_region_info *> get_all_regions();

    // Clear all allocations
    void clear();

private:
    rdma_connection * conn_;
    std::mutex mutex_;

    // Allocated regions (we own the memory)
    std::vector<std::unique_ptr<memory_region_info>> allocated_;

    // Registered regions (we don't own the memory)
    std::vector<std::unique_ptr<memory_region_info>> registered_;

    // Map remote_ptr to region for fast lookup
    std::unordered_map<uint64_t, memory_region_info *> remote_ptr_map_;

    // Internal allocation helpers
    void * alloc_host(size_t size);
    void * alloc_host_pinned(size_t size);
    void free_host(void * ptr);
    void free_host_pinned(void * ptr);
};

// Staging buffer for RDMA operations
// Used when source/destination memory is not directly RDMA-registered
class rdma_staging_buffer {
public:
    rdma_staging_buffer(rdma_memory_pool * pool, size_t initial_size = 1024 * 1024);
    ~rdma_staging_buffer();

    // Disable copy
    rdma_staging_buffer(const rdma_staging_buffer&) = delete;
    rdma_staging_buffer& operator=(const rdma_staging_buffer&) = delete;

    // Get buffer for size bytes, reallocating if necessary
    // Returns pointer to buffer data and sets mr to the memory region
    void * get_buffer(size_t size, struct ibv_mr ** mr);

    // Get current buffer info
    memory_region_info * get_info() const { return info_; }

private:
    rdma_memory_pool * pool_;
    memory_region_info * info_;
    size_t current_size_;
};

} // namespace ggml_rdma
