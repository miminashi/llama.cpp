#include "rdma-gdr.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#ifdef GGML_RDMA_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

static const char * RDMA_DEBUG = std::getenv("GGML_RDMA_DEBUG");

#define RDMA_LOG_DBG(...) \
    do { if (RDMA_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

namespace ggml_rdma {

// Check if nvidia-peermem module is loaded
bool gdr_memory_manager::is_peermem_loaded() {
    // Check /sys/module/nvidia_peermem
    std::ifstream module_file("/sys/module/nvidia_peermem/initstate");
    if (module_file.is_open()) {
        std::string state;
        std::getline(module_file, state);
        return state == "live";
    }

    // Alternative: check /proc/modules
    std::ifstream proc_modules("/proc/modules");
    if (proc_modules.is_open()) {
        std::string line;
        while (std::getline(proc_modules, line)) {
            if (line.find("nvidia_peermem") != std::string::npos) {
                return true;
            }
        }
    }

    return false;
}

bool gdr_memory_manager::check_peermem_support() {
#ifdef GGML_RDMA_CUDA
    // Check if CUDA is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        RDMA_LOG_DBG("[gdr_memory_manager] No CUDA devices found\n");
        return false;
    }

    // Check nvidia-peermem
    if (!is_peermem_loaded()) {
        RDMA_LOG_DBG("[gdr_memory_manager] nvidia-peermem module not loaded\n");
        return false;
    }

    return true;
#else
    return false;
#endif
}

bool gdr_memory_manager::is_available() {
    static bool checked = false;
    static bool available = false;

    if (!checked) {
        // Check if GPUDirect is explicitly disabled via environment variable
        const char * no_gdr = std::getenv("GGML_RDMA_NO_GDR");
        if (no_gdr && (std::string(no_gdr) == "1" || std::string(no_gdr) == "true")) {
            available = false;
            checked = true;
            GGML_LOG_INFO("[gdr_memory_manager] GPUDirect RDMA disabled via GGML_RDMA_NO_GDR\n");
            return available;
        }

        available = check_peermem_support();
        checked = true;

        if (available) {
            GGML_LOG_INFO("[gdr_memory_manager] GPUDirect RDMA is available\n");
        } else {
            GGML_LOG_DEBUG("[gdr_memory_manager] GPUDirect RDMA is not available\n");
        }
    }

    return available;
}

// gdr_memory_manager implementation

gdr_memory_manager::gdr_memory_manager(rdma_connection * conn)
    : conn_(conn) {
}

gdr_memory_manager::~gdr_memory_manager() {
    clear();
}

bool gdr_memory_manager::init_device(int device_id) {
#ifdef GGML_RDMA_CUDA
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if already initialized
    for (int id : initialized_devices_) {
        if (id == device_id) {
            return true;
        }
    }

    // Set CUDA device
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("[gdr_memory_manager] Failed to set CUDA device %d: %s\n",
                       device_id, cudaGetErrorString(err));
        return false;
    }

    // Check device properties for GPUDirect support
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("[gdr_memory_manager] Failed to get device properties: %s\n",
                       cudaGetErrorString(err));
        return false;
    }

    // GPUDirect RDMA requires compute capability 3.5+
    int compute_cap = props.major * 10 + props.minor;
    if (compute_cap < 35) {
        GGML_LOG_ERROR("[gdr_memory_manager] GPU device %d compute capability %d.%d < 3.5, "
                       "GPUDirect RDMA not supported\n",
                       device_id, props.major, props.minor);
        return false;
    }

    initialized_devices_.push_back(device_id);
    GGML_LOG_INFO("[gdr_memory_manager] Initialized GPU device %d (%s) for GPUDirect RDMA\n",
                  device_id, props.name);
    return true;
#else
    GGML_UNUSED(device_id);
    GGML_LOG_ERROR("[gdr_memory_manager] CUDA support not compiled\n");
    return false;
#endif
}

bool gdr_memory_manager::is_device_initialized(int device_id) const {
    for (int id : initialized_devices_) {
        if (id == device_id) {
            return true;
        }
    }
    return false;
}

gpu_memory_region * gdr_memory_manager::register_gpu_memory(void * gpu_addr, size_t size, int device_id) {
#ifdef GGML_RDMA_CUDA
    std::lock_guard<std::mutex> lock(mutex_);

    if (!is_device_initialized(device_id)) {
        GGML_LOG_ERROR("[gdr_memory_manager] Device %d not initialized\n", device_id);
        return nullptr;
    }

    // Set CUDA device
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("[gdr_memory_manager] Failed to set CUDA device: %s\n",
                       cudaGetErrorString(err));
        return nullptr;
    }

    // Get CUDA driver handle for the memory
    CUdeviceptr d_ptr = (CUdeviceptr)gpu_addr;

    // Register GPU memory with RDMA
    // nvidia-peermem allows direct registration of GPU memory
    int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    struct ibv_mr * mr = conn_->register_memory(gpu_addr, size, access_flags);
    if (!mr) {
        GGML_LOG_ERROR("[gdr_memory_manager] Failed to register GPU memory with RDMA\n");
        return nullptr;
    }

    auto region = std::make_unique<gpu_memory_region>();
    region->gpu_addr = gpu_addr;
    region->host_addr = nullptr;
    region->size = size;
    region->mr = mr;
    region->device_id = device_id;
    region->peer_mapped = true;

    gpu_memory_region * result = region.get();
    addr_map_[gpu_addr] = result;
    regions_.push_back(std::move(region));

    RDMA_LOG_DBG("[gdr_memory_manager] Registered GPU memory: addr=%p, size=%zu, device=%d\n",
                 gpu_addr, size, device_id);
    return result;
#else
    GGML_UNUSED(gpu_addr);
    GGML_UNUSED(size);
    GGML_UNUSED(device_id);
    GGML_LOG_ERROR("[gdr_memory_manager] CUDA support not compiled\n");
    return nullptr;
#endif
}

void gdr_memory_manager::deregister_gpu_memory(gpu_memory_region * region) {
    if (!region) return;

    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = regions_.begin(); it != regions_.end(); ++it) {
        if (it->get() == region) {
            // Deregister from RDMA
            if (region->mr) {
                conn_->deregister_memory(region->mr);
            }

            addr_map_.erase(region->gpu_addr);
            regions_.erase(it);

            RDMA_LOG_DBG("[gdr_memory_manager] Deregistered GPU memory at %p\n", region->gpu_addr);
            return;
        }
    }

    GGML_LOG_WARN("[gdr_memory_manager] Attempted to deregister unknown GPU memory region\n");
}

gpu_memory_region * gdr_memory_manager::alloc_gpu_memory(size_t size, int device_id) {
#ifdef GGML_RDMA_CUDA
    // Set device
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("[gdr_memory_manager] Failed to set CUDA device: %s\n",
                       cudaGetErrorString(err));
        return nullptr;
    }

    // Allocate GPU memory
    void * gpu_addr = nullptr;
    err = cudaMalloc(&gpu_addr, size);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("[gdr_memory_manager] Failed to allocate GPU memory: %s\n",
                       cudaGetErrorString(err));
        return nullptr;
    }

    // Register for RDMA
    gpu_memory_region * region = register_gpu_memory(gpu_addr, size, device_id);
    if (!region) {
        cudaFree(gpu_addr);
        return nullptr;
    }

    RDMA_LOG_DBG("[gdr_memory_manager] Allocated GPU memory: addr=%p, size=%zu, device=%d\n",
                 gpu_addr, size, device_id);
    return region;
#else
    GGML_UNUSED(size);
    GGML_UNUSED(device_id);
    GGML_LOG_ERROR("[gdr_memory_manager] CUDA support not compiled\n");
    return nullptr;
#endif
}

void gdr_memory_manager::free_gpu_memory(gpu_memory_region * region) {
#ifdef GGML_RDMA_CUDA
    if (!region) return;

    void * gpu_addr = region->gpu_addr;

    // Deregister first
    deregister_gpu_memory(region);

    // Free GPU memory
    cudaFree(gpu_addr);

    RDMA_LOG_DBG("[gdr_memory_manager] Freed GPU memory at %p\n", gpu_addr);
#else
    GGML_UNUSED(region);
#endif
}

gpu_memory_region * gdr_memory_manager::get_region(void * gpu_addr) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = addr_map_.find(gpu_addr);
    if (it != addr_map_.end()) {
        return it->second;
    }
    return nullptr;
}

remote_memory_info gdr_memory_manager::get_remote_info(gpu_memory_region * region) {
    remote_memory_info info = {};
    if (region && region->mr) {
        info.addr = (uint64_t)region->mr->addr;
        info.rkey = region->mr->rkey;
        info.size = region->size;
    }
    return info;
}

void gdr_memory_manager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

#ifdef GGML_RDMA_CUDA
    for (auto & region : regions_) {
        if (region->mr) {
            conn_->deregister_memory(region->mr);
        }
        // Note: we don't free GPU memory here as we may not own it
    }
#endif

    regions_.clear();
    addr_map_.clear();

    RDMA_LOG_DBG("[gdr_memory_manager] Cleared all GPU memory registrations\n");
}

// pinned_memory_manager implementation

pinned_memory_manager::pinned_memory_manager()
    : staging_buffer_(nullptr), staging_buffer_size_(0) {
}

pinned_memory_manager::~pinned_memory_manager() {
    // Free all allocations
    for (auto & [ptr, size] : allocations_) {
#ifdef GGML_RDMA_CUDA
        cudaFreeHost(ptr);
#else
        ::free(ptr);
#endif
    }
    allocations_.clear();

    // Free staging buffer
    if (staging_buffer_) {
#ifdef GGML_RDMA_CUDA
        cudaFreeHost(staging_buffer_);
#else
        ::free(staging_buffer_);
#endif
        staging_buffer_ = nullptr;
    }
}

void * pinned_memory_manager::alloc(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    void * ptr = nullptr;

#ifdef GGML_RDMA_CUDA
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("[pinned_memory_manager] Failed to allocate pinned memory: %s\n",
                       cudaGetErrorString(err));
        return nullptr;
    }
#else
    if (posix_memalign(&ptr, 4096, size) != 0) {
        GGML_LOG_ERROR("[pinned_memory_manager] Failed to allocate aligned memory\n");
        return nullptr;
    }
#endif

    allocations_.push_back({ptr, size});
    RDMA_LOG_DBG("[pinned_memory_manager] Allocated %zu bytes at %p\n", size, ptr);
    return ptr;
}

void pinned_memory_manager::free(void * ptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = allocations_.begin(); it != allocations_.end(); ++it) {
        if (it->first == ptr) {
#ifdef GGML_RDMA_CUDA
            cudaFreeHost(ptr);
#else
            ::free(ptr);
#endif
            allocations_.erase(it);
            RDMA_LOG_DBG("[pinned_memory_manager] Freed memory at %p\n", ptr);
            return;
        }
    }

    GGML_LOG_WARN("[pinned_memory_manager] Attempted to free unknown pointer\n");
}

void * pinned_memory_manager::get_staging_buffer(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (staging_buffer_ && staging_buffer_size_ >= size) {
        return staging_buffer_;
    }

    // Need to reallocate
    if (staging_buffer_) {
#ifdef GGML_RDMA_CUDA
        cudaFreeHost(staging_buffer_);
#else
        ::free(staging_buffer_);
#endif
    }

    // Allocate with growth factor
    size_t new_size = size + size / 4;

#ifdef GGML_RDMA_CUDA
    cudaError_t err = cudaMallocHost(&staging_buffer_, new_size);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("[pinned_memory_manager] Failed to allocate staging buffer: %s\n",
                       cudaGetErrorString(err));
        staging_buffer_ = nullptr;
        staging_buffer_size_ = 0;
        return nullptr;
    }
#else
    if (posix_memalign(&staging_buffer_, 4096, new_size) != 0) {
        staging_buffer_ = nullptr;
        staging_buffer_size_ = 0;
        return nullptr;
    }
#endif

    staging_buffer_size_ = new_size;
    RDMA_LOG_DBG("[pinned_memory_manager] Allocated staging buffer: %zu bytes\n", new_size);
    return staging_buffer_;
}

} // namespace ggml_rdma
