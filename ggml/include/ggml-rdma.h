#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RDMA_PROTO_MAJOR_VERSION    1
#define RDMA_PROTO_MINOR_VERSION    2
#define RDMA_PROTO_PATCH_VERSION    0
#define GGML_RDMA_MAX_SERVERS       16

// Backend API

// Initialize RDMA backend for a remote server
// endpoint: server address in format "host:port"
// device: device index on the remote server
GGML_BACKEND_API ggml_backend_t ggml_backend_rdma_init(const char * endpoint, uint32_t device);

// Check if a backend is RDMA backend
GGML_BACKEND_API bool ggml_backend_is_rdma(ggml_backend_t backend);

// Get buffer type for RDMA backend
// endpoint: server address in format "host:port"
// device: device index on the remote server
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rdma_buffer_type(const char * endpoint, uint32_t device);

// Get device memory information from remote server
GGML_BACKEND_API void ggml_backend_rdma_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

// Start RDMA server
// endpoint: bind address in format "host:port"
// cache_dir: optional directory for caching tensor data (can be NULL)
// n_threads: number of threads for backend
// n_devices: number of devices to expose
// devices: array of backend devices to expose
GGML_BACKEND_API void ggml_backend_rdma_start_server(const char * endpoint, const char * cache_dir,
                                                      size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

// Backend registration
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rdma_reg(void);

// Add a remote RDMA server to the registry
// Returns the registry for the added server
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rdma_add_server(const char * endpoint);

// GPUDirect RDMA configuration

// Enable GPUDirect RDMA for the backend (requires nvidia-peermem)
// Returns true if GPUDirect was successfully enabled
GGML_BACKEND_API bool ggml_backend_rdma_enable_gdr(ggml_backend_t backend);

// Check if GPUDirect RDMA is available on the system
GGML_BACKEND_API bool ggml_backend_rdma_gdr_available(void);

// RDMA transport statistics

typedef struct {
    uint64_t bytes_sent;        // Total bytes sent via RDMA Write
    uint64_t bytes_received;    // Total bytes received via RDMA Read
    uint64_t rdma_writes;       // Number of RDMA Write operations
    uint64_t rdma_reads;        // Number of RDMA Read operations
    uint64_t send_ops;          // Number of Send operations (control messages)
    uint64_t recv_ops;          // Number of Receive operations (control messages)
} ggml_rdma_stats_t;

// Get RDMA transport statistics
GGML_BACKEND_API void ggml_backend_rdma_get_stats(ggml_backend_t backend, ggml_rdma_stats_t * stats);

// Reset RDMA transport statistics
GGML_BACKEND_API void ggml_backend_rdma_reset_stats(ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
