#include "rdma-memory.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

#ifdef GGML_RDMA_CUDA
#include <cuda_runtime.h>
#endif

static const char * RDMA_DEBUG = std::getenv("GGML_RDMA_DEBUG");

#define RDMA_LOG_DBG(...) \
    do { if (RDMA_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

namespace ggml_rdma {

// rdma_memory_pool implementation

rdma_memory_pool::rdma_memory_pool(rdma_connection * conn)
    : conn_(conn) {
}

rdma_memory_pool::~rdma_memory_pool() {
    clear();
}

void * rdma_memory_pool::alloc_host(size_t size) {
    // Allocate page-aligned memory for better RDMA performance
    void * ptr = nullptr;
    if (posix_memalign(&ptr, 4096, size) != 0) {
        return nullptr;
    }
    return ptr;
}

void * rdma_memory_pool::alloc_host_pinned(size_t size) {
#ifdef GGML_RDMA_CUDA
    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("[rdma_memory_pool] Failed to allocate pinned memory: %s\n",
                       cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
#else
    // Fall back to mlock'd memory
    void * ptr = alloc_host(size);
    if (ptr && mlock(ptr, size) != 0) {
        GGML_LOG_WARN("[rdma_memory_pool] Failed to lock memory, using unlocked\n");
    }
    return ptr;
#endif
}

void rdma_memory_pool::free_host(void * ptr) {
    ::free(ptr);
}

void rdma_memory_pool::free_host_pinned(void * ptr) {
#ifdef GGML_RDMA_CUDA
    cudaFreeHost(ptr);
#else
    // Note: munlock is called automatically when memory is freed
    ::free(ptr);
#endif
}

memory_region_info * rdma_memory_pool::alloc(size_t size, alloc_type type) {
    std::lock_guard<std::mutex> lock(mutex_);

    void * ptr = nullptr;

    switch (type) {
        case alloc_type::HOST:
            ptr = alloc_host(size);
            break;
        case alloc_type::HOST_PINNED:
            ptr = alloc_host_pinned(size);
            break;
        case alloc_type::GPU_DIRECT:
            // GPU memory allocation is handled by rdma-gdr
            GGML_LOG_ERROR("[rdma_memory_pool] GPU_DIRECT allocation should use gdr_memory_manager\n");
            return nullptr;
    }

    if (!ptr) {
        GGML_LOG_ERROR("[rdma_memory_pool] Failed to allocate %zu bytes\n", size);
        return nullptr;
    }

    // Register with RDMA
    int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    struct ibv_mr * mr = conn_->register_memory(ptr, size, access_flags);
    if (!mr) {
        if (type == alloc_type::HOST_PINNED) {
            free_host_pinned(ptr);
        } else {
            free_host(ptr);
        }
        return nullptr;
    }

    auto info = std::make_unique<memory_region_info>();
    info->addr = ptr;
    info->size = size;
    info->mr = mr;
    info->is_gpu = false;
    info->gpu_device = -1;
    info->remote_ptr = 0;
    info->type = type;  // Track allocation type for proper deallocation

    memory_region_info * result = info.get();
    allocated_.push_back(std::move(info));

    RDMA_LOG_DBG("[rdma_memory_pool] Allocated %zu bytes at %p (type=%d)\n", size, ptr, (int)type);
    return result;
}

void rdma_memory_pool::free(memory_region_info * info) {
    RDMA_LOG_DBG("[rdma_memory_pool::free] Entering, info=%p\n", (void*)info);
    if (!info) return;

    RDMA_LOG_DBG("[rdma_memory_pool::free] Taking lock\n");
    std::lock_guard<std::mutex> lock(mutex_);
    RDMA_LOG_DBG("[rdma_memory_pool::free] Lock acquired, searching in %zu entries\n", allocated_.size());

    for (auto it = allocated_.begin(); it != allocated_.end(); ++it) {
        if (it->get() == info) {
            RDMA_LOG_DBG("[rdma_memory_pool::free] Found entry, mr=%p\n", (void*)info->mr);
            // Deregister from RDMA
            if (info->mr) {
                RDMA_LOG_DBG("[rdma_memory_pool::free] Deregistering MR\n");
                conn_->deregister_memory(info->mr);
                RDMA_LOG_DBG("[rdma_memory_pool::free] MR deregistered\n");
            }

            // Free memory based on allocation type
            RDMA_LOG_DBG("[rdma_memory_pool::free] Freeing memory at %p (type=%d)\n", info->addr, (int)info->type);
            switch (info->type) {
                case alloc_type::HOST:
                    free_host(info->addr);
                    break;
                case alloc_type::HOST_PINNED:
                    free_host_pinned(info->addr);
                    break;
                case alloc_type::GPU_DIRECT:
                    // GPU memory is handled by gdr_memory_manager
                    break;
            }
            RDMA_LOG_DBG("[rdma_memory_pool::free] Memory freed\n");

            // Remove from remote_ptr_map if present
            if (info->remote_ptr != 0) {
                remote_ptr_map_.erase(info->remote_ptr);
            }

            RDMA_LOG_DBG("[rdma_memory_pool::free] Erasing from allocated_\n");
            allocated_.erase(it);
            RDMA_LOG_DBG("[rdma_memory_pool] Freed memory at %p\n", info->addr);
            return;
        }
    }

    GGML_LOG_WARN("[rdma_memory_pool] Attempted to free unknown memory region\n");
}

memory_region_info * rdma_memory_pool::register_memory(void * addr, size_t size, int access_flags) {
    std::lock_guard<std::mutex> lock(mutex_);

    struct ibv_mr * mr = conn_->register_memory(addr, size, access_flags);
    if (!mr) {
        return nullptr;
    }

    auto info = std::make_unique<memory_region_info>();
    info->addr = addr;
    info->size = size;
    info->mr = mr;
    info->is_gpu = false;
    info->gpu_device = -1;
    info->remote_ptr = 0;

    memory_region_info * result = info.get();
    registered_.push_back(std::move(info));

    RDMA_LOG_DBG("[rdma_memory_pool] Registered %zu bytes at %p\n", size, addr);
    return result;
}

void rdma_memory_pool::deregister_memory(memory_region_info * info) {
    if (!info) return;

    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = registered_.begin(); it != registered_.end(); ++it) {
        if (it->get() == info) {
            if (info->mr) {
                conn_->deregister_memory(info->mr);
            }

            if (info->remote_ptr != 0) {
                remote_ptr_map_.erase(info->remote_ptr);
            }

            registered_.erase(it);
            RDMA_LOG_DBG("[rdma_memory_pool] Deregistered memory at %p\n", info->addr);
            return;
        }
    }

    GGML_LOG_WARN("[rdma_memory_pool] Attempted to deregister unknown memory region\n");
}

memory_region_info * rdma_memory_pool::get_region(uint64_t remote_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = remote_ptr_map_.find(remote_ptr);
    if (it != remote_ptr_map_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<memory_region_info *> rdma_memory_pool::get_all_regions() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<memory_region_info *> result;
    for (auto & info : allocated_) {
        result.push_back(info.get());
    }
    for (auto & info : registered_) {
        result.push_back(info.get());
    }
    return result;
}

void rdma_memory_pool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Free all allocated regions
    for (auto & info : allocated_) {
        if (info->mr) {
            conn_->deregister_memory(info->mr);
        }
        if (!info->is_gpu) {
            free_host(info->addr);
        }
    }
    allocated_.clear();

    // Deregister all registered regions (but don't free memory)
    for (auto & info : registered_) {
        if (info->mr) {
            conn_->deregister_memory(info->mr);
        }
    }
    registered_.clear();

    remote_ptr_map_.clear();

    RDMA_LOG_DBG("[rdma_memory_pool] Cleared all regions\n");
}

// rdma_staging_buffer implementation

rdma_staging_buffer::rdma_staging_buffer(rdma_memory_pool * pool, size_t initial_size)
    : pool_(pool), info_(nullptr), current_size_(0) {

    info_ = pool_->alloc(initial_size, alloc_type::HOST_PINNED);
    if (info_) {
        current_size_ = initial_size;
    }
}

rdma_staging_buffer::~rdma_staging_buffer() {
    if (info_) {
        pool_->free(info_);
    }
}

void * rdma_staging_buffer::get_buffer(size_t size, struct ibv_mr ** mr) {
    if (size > current_size_) {
        // Need to reallocate
        if (info_) {
            pool_->free(info_);
        }

        // Allocate with some growth factor
        size_t new_size = size + size / 4; // 25% extra
        info_ = pool_->alloc(new_size, alloc_type::HOST_PINNED);
        if (!info_) {
            current_size_ = 0;
            *mr = nullptr;
            return nullptr;
        }
        current_size_ = new_size;
    }

    *mr = info_->mr;
    return info_->addr;
}

} // namespace ggml_rdma
