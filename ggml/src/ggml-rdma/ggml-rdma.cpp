#include "ggml-rdma.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"

#include "rdma-transport.h"
#include "rdma-memory.h"
#include "rdma-gdr.h"

#include <cinttypes>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <algorithm>

using namespace ggml_rdma;

static const char * RDMA_DEBUG = std::getenv("GGML_RDMA_DEBUG");

#define RDMA_LOG_DBG(...) \
    do { if (RDMA_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

// Ensure response is valid
#define RDMA_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RDMA server crashed or returned malformed response")

// Row padding for quantized tensors (must match server-side value)
#define RDMA_MATRIX_ROW_PADDING 512

// All RDMA structures must be packed for wire protocol
#pragma pack(push, 1)

// Tensor serialization (compatible with RPC for potential interop)
struct rdma_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];
    char padding[4];
};

static_assert(sizeof(rdma_tensor) % 8 == 0, "rdma_tensor size must be multiple of 8");

// RDMA commands
enum rdma_cmd : uint8_t {
    RDMA_CMD_ALLOC_BUFFER = 0,
    RDMA_CMD_GET_ALIGNMENT,
    RDMA_CMD_GET_MAX_SIZE,
    RDMA_CMD_BUFFER_GET_BASE,
    RDMA_CMD_FREE_BUFFER,
    RDMA_CMD_BUFFER_CLEAR,
    RDMA_CMD_SET_TENSOR,
    RDMA_CMD_GET_TENSOR,
    RDMA_CMD_COPY_TENSOR,
    RDMA_CMD_GRAPH_COMPUTE,
    RDMA_CMD_GET_DEVICE_MEMORY,
    RDMA_CMD_INIT_TENSOR,
    RDMA_CMD_GET_ALLOC_SIZE,
    RDMA_CMD_HELLO,
    RDMA_CMD_DEVICE_COUNT,
    RDMA_CMD_GRAPH_RECOMPUTE,
    // RDMA-specific commands
    RDMA_CMD_REGISTER_MR,      // Register memory region for RDMA
    RDMA_CMD_DEREGISTER_MR,    // Deregister memory region
    RDMA_CMD_GET_MR_INFO,      // Get memory region info for RDMA ops
    RDMA_CMD_RDMA_WRITE_DONE,  // Notify completion of RDMA write
    RDMA_CMD_RDMA_READ_DONE,   // Notify completion of RDMA read
    RDMA_CMD_COUNT,
};

static_assert(RDMA_CMD_HELLO == 13, "RDMA_CMD_HELLO must be 13 for RPC compatibility check");

struct rdma_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t gdr_available;  // GPUDirect RDMA available
};

struct rdma_msg_device_count_rsp {
    uint32_t device_count;
};

struct rdma_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rdma_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
    // RDMA memory region info for direct access
    uint64_t mr_addr;
    uint32_t mr_rkey;
};

struct rdma_msg_get_alignment_req {
    uint32_t device;
};

struct rdma_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rdma_msg_get_max_size_req {
    uint32_t device;
};

struct rdma_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rdma_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rdma_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rdma_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rdma_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rdma_msg_get_tensor_req {
    rdma_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rdma_msg_copy_tensor_req {
    rdma_tensor src;
    rdma_tensor dst;
};

struct rdma_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rdma_msg_get_device_memory_req {
    uint32_t device;
};

struct rdma_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rdma_msg_init_tensor_req {
    rdma_tensor tensor;
};

struct rdma_msg_get_alloc_size_req {
    uint32_t   device;
    rdma_tensor tensor;
    rdma_tensor srcs[GGML_MAX_SRC];
};

struct rdma_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rdma_msg_graph_recompute_req {
    uint32_t device;
};

// RDMA-specific message structures
struct rdma_msg_register_mr_req {
    uint64_t remote_ptr;  // Buffer pointer
    uint64_t size;
};

struct rdma_msg_register_mr_rsp {
    uint64_t mr_addr;
    uint32_t mr_rkey;
    uint8_t  success;
};

struct rdma_msg_get_mr_info_req {
    uint64_t remote_ptr;
};

struct rdma_msg_get_mr_info_rsp {
    uint64_t mr_addr;
    uint32_t mr_rkey;
    uint64_t size;
};

#pragma pack(pop)

// RDMA data structures

static ggml_guid_t ggml_backend_rdma_guid() {
    static ggml_guid guid = {0x52, 0x44, 0x4d, 0x41, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

// Maximum entries in alloc_size cache
static constexpr size_t RDMA_ALLOC_SIZE_CACHE_MAX_ENTRIES = 1000;

struct alloc_size_cache_t {
    std::unordered_map<uint64_t, size_t> entries;
    mutable std::mutex mutex;
};

struct ggml_backend_rdma_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;

    // Cache for get_alloc_size responses (FLASH_ATTN_EXT, MUL_MAT_ID)
    alloc_size_cache_t alloc_size_cache;
};

struct ggml_backend_rdma_buffer_context {
    std::shared_ptr<rdma_connection> conn;
    void * base_ptr;
    uint64_t remote_ptr;
    // RDMA MR info for direct transfers
    uint64_t mr_addr;
    uint32_t mr_rkey;
    size_t   size;
    // Local memory pool for staging
    std::unique_ptr<rdma_memory_pool> mem_pool;
};

struct graph_cache {
    bool is_cached(const ggml_cgraph * cgraph) {
        if ((int)last_graph.size() != cgraph->n_nodes) {
            return false;
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (memcmp(&last_graph[i], cgraph->nodes[i], sizeof(ggml_tensor)) != 0) {
                return false;
            }
        }
        return true;
    }

    void add(const ggml_cgraph * cgraph) {
        last_graph.resize(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            memcpy(&last_graph[i], cgraph->nodes[i], sizeof(ggml_tensor));
        }
    }

    std::vector<ggml_tensor> last_graph;
};

struct ggml_backend_rdma_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    graph_cache gc;
    std::shared_ptr<rdma_connection> conn;
    bool        gdr_enabled;
    rdma_stats  stats;
};

// Connection management

static rdma_connection_manager & get_connection_manager() {
    static rdma_connection_manager manager;
    return manager;
}

static std::shared_ptr<rdma_connection> get_connection(const std::string & endpoint) {
    return get_connection_manager().get_connection(endpoint);
}

// Message sending helpers

// Internal: send command and input data only (no response handling)
static bool send_rdma_cmd_raw(rdma_connection * conn, rdma_cmd cmd, const void * input, size_t input_size,
                              struct ibv_mr * send_mr) {
    // Protocol: | cmd (1 byte) | input_size (8 bytes) | input_data |
    // Send each part separately to match server's receive pattern
    // Note: send_mr is only valid for input data, not for header fields
    uint8_t cmd_byte = static_cast<uint8_t>(cmd);
    if (!conn->send(&cmd_byte, 1, nullptr)) {
        return false;
    }

    uint64_t size = input_size;
    if (!conn->send(&size, sizeof(size), nullptr)) {
        return false;
    }

    if (input_size > 0) {
        if (!conn->send(input, input_size, send_mr)) {
            return false;
        }
    }

    return true;
}

// Send command with no response data expected (server still sends rsp_size=0)
static bool send_rdma_cmd(rdma_connection * conn, rdma_cmd cmd, const void * input, size_t input_size,
                          struct ibv_mr * send_mr) {
    if (!send_rdma_cmd_raw(conn, cmd, input, input_size, send_mr)) {
        return false;
    }

    // Server always sends rsp_size (8 bytes), even for commands with no response data.
    // We must receive it to keep the protocol synchronized.
    uint64_t rsp_size = 0;
    if (!conn->recv(&rsp_size, sizeof(rsp_size), nullptr)) {
        return false;
    }
    // rsp_size should be 0 for commands without response data
    if (rsp_size != 0) {
        GGML_LOG_ERROR("[rdma] Unexpected response size for no-response command: %" PRIu64 "\n", rsp_size);
        return false;
    }

    return true;
}

static bool recv_rdma_rsp(rdma_connection * conn, void * output, size_t output_size,
                          struct ibv_mr * recv_mr) {
    // Response: | output_size (8 bytes) | output_data |
    uint64_t size = 0;
    if (!conn->recv(&size, sizeof(size), recv_mr)) {
        return false;
    }
    if (size != output_size) {
        GGML_LOG_ERROR("[rdma] Response size mismatch: expected %zu, got %" PRIu64 "\n", output_size, size);
        return false;
    }
    if (output_size > 0) {
        return conn->recv(output, output_size, recv_mr);
    }
    return true;
}

static bool send_rdma_cmd_with_rsp(rdma_connection * conn, rdma_cmd cmd,
                                    const void * input, size_t input_size,
                                    void * output, size_t output_size) {
    if (!send_rdma_cmd_raw(conn, cmd, input, input_size, nullptr)) {
        return false;
    }
    return recv_rdma_rsp(conn, output, output_size, nullptr);
}

// Version check

static bool check_server_version(rdma_connection * conn) {
    rdma_msg_hello_rsp response;
    bool status = send_rdma_cmd_with_rsp(conn, RDMA_CMD_HELLO, nullptr, 0, &response, sizeof(response));
    RDMA_STATUS_ASSERT(status);
    if (response.major != RDMA_PROTO_MAJOR_VERSION || response.minor > RDMA_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RDMA server version mismatch: %d.%d.%d\n", response.major, response.minor, response.patch);
        return false;
    }
    if (response.minor != RDMA_PROTO_MINOR_VERSION || response.patch != RDMA_PROTO_PATCH_VERSION) {
        GGML_LOG_INFO("WARNING: RDMA server version mismatch: %d.%d.%d\n", response.major, response.minor, response.patch);
    }
    return true;
}

// Tensor serialization

static rdma_tensor serialize_tensor(const ggml_tensor * tensor) {
    rdma_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;

    // Check if buffer is RDMA buffer
    if (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        // Check buffer type via comparing free function pointer
        if (buffer->context) {
            ggml_backend_rdma_buffer_context * ctx = (ggml_backend_rdma_buffer_context *)buffer->context;
            result.buffer = ctx->remote_ptr;
        } else {
            result.buffer = 0;
        }
    } else {
        result.buffer = 0;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data = reinterpret_cast<uint64_t>(tensor->data);

    memset(result.name, 0, sizeof(result.name));
    memset(result.padding, 0, sizeof(result.padding));
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);

    return result;
}

// Buffer interface

static void ggml_backend_rdma_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rdma_buffer_context * ctx = (ggml_backend_rdma_buffer_context *)buffer->context;
    rdma_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rdma_cmd(ctx->conn.get(), RDMA_CMD_FREE_BUFFER, &request, sizeof(request), nullptr);
    RDMA_STATUS_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rdma_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rdma_buffer_context * ctx = (ggml_backend_rdma_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rdma_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rdma_msg_buffer_get_base_rsp response;
    bool status = send_rdma_cmd_with_rsp(ctx->conn.get(), RDMA_CMD_BUFFER_GET_BASE,
                                          &request, sizeof(request), &response, sizeof(response));
    RDMA_STATUS_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static enum ggml_status ggml_backend_rdma_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rdma_buffer_context * ctx = (ggml_backend_rdma_buffer_context *)buffer->context;

    // Only init quantized tensors that need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rdma_msg_init_tensor_req request;
        request.tensor = serialize_tensor(tensor);
        bool status = send_rdma_cmd(ctx->conn.get(), RDMA_CMD_INIT_TENSOR, &request, sizeof(request), nullptr);
        RDMA_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rdma_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                 const void * data, size_t offset, size_t size) {
    ggml_backend_rdma_buffer_context * ctx = (ggml_backend_rdma_buffer_context *)buffer->context;

    // Ensure base_ptr is initialized
    if (ctx->base_ptr == nullptr) {
        ggml_backend_rdma_buffer_get_base(buffer);
    }

    // Try RDMA write if we have MR info and base_ptr is valid
    if (ctx->mr_rkey != 0 && ctx->mem_pool && ctx->base_ptr != nullptr) {
        // Allocate staging buffer and register for RDMA
        memory_region_info * staging = ctx->mem_pool->alloc(size, alloc_type::HOST_PINNED);
        if (staging) {
            // Copy data to staging buffer
            memcpy(staging->addr, data, size);

            // Calculate remote address
            remote_memory_info remote;
            remote.addr = ctx->mr_addr + (reinterpret_cast<uint64_t>(tensor->data) - reinterpret_cast<uint64_t>(ctx->base_ptr)) + offset;
            remote.rkey = ctx->mr_rkey;
            remote.size = size;

            // Perform RDMA write
            RDMA_LOG_DBG("[rdma_set_tensor] About to RDMA write: remote.addr=0x%lx, base_ptr=%p, tensor->data=%p\n",
                         remote.addr, ctx->base_ptr, tensor->data);
            if (ctx->conn->rdma_write(staging->addr, size, staging->mr, remote, true)) {
                RDMA_LOG_DBG("[rdma_set_tensor] RDMA write succeeded, freeing staging\n");
                ctx->mem_pool->free(staging);
                RDMA_LOG_DBG("[rdma_set_tensor] Staging freed, returning\n");
                return;
            }
            RDMA_LOG_DBG("[rdma_set_tensor] RDMA write failed, freeing staging and falling back\n");
            ctx->mem_pool->free(staging);
        }
    }

    // Fall back to send-based transfer
    rdma_tensor rpc_tensor = serialize_tensor(tensor);
    size_t input_size = sizeof(rdma_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);

    bool status = send_rdma_cmd(ctx->conn.get(), RDMA_CMD_SET_TENSOR, input.data(), input.size(), nullptr);
    RDMA_STATUS_ASSERT(status);
}

static void ggml_backend_rdma_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                 void * data, size_t offset, size_t size) {
    ggml_backend_rdma_buffer_context * ctx = (ggml_backend_rdma_buffer_context *)buffer->context;

    // Ensure base_ptr is initialized
    if (ctx->base_ptr == nullptr) {
        ggml_backend_rdma_buffer_get_base(const_cast<ggml_backend_buffer_t>(buffer));
    }

    // Try RDMA read if we have MR info and base_ptr is valid
    if (ctx->mr_rkey != 0 && ctx->mem_pool && ctx->base_ptr != nullptr) {
        memory_region_info * staging = ctx->mem_pool->alloc(size, alloc_type::HOST_PINNED);
        if (staging) {
            remote_memory_info remote;
            remote.addr = ctx->mr_addr + (reinterpret_cast<uint64_t>(tensor->data) - reinterpret_cast<uint64_t>(ctx->base_ptr)) + offset;
            remote.rkey = ctx->mr_rkey;
            remote.size = size;

            if (ctx->conn->rdma_read(staging->addr, size, staging->mr, remote, true)) {
                memcpy(data, staging->addr, size);
                ctx->mem_pool->free(staging);
                RDMA_LOG_DBG("[rdma] RDMA read: %zu bytes from tensor %s\n", size, tensor->name);
                return;
            }

            ctx->mem_pool->free(staging);
        }
    }

    // Fall back to recv-based transfer
    rdma_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    bool status = send_rdma_cmd_with_rsp(ctx->conn.get(), RDMA_CMD_GET_TENSOR,
                                          &request, sizeof(request), data, size);
    RDMA_STATUS_ASSERT(status);
}

static bool ggml_backend_rdma_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // Check if both tensors are on RDMA buffers with same connection
    if (!src->buffer || !dst->buffer) {
        return false;
    }

    ggml_backend_rdma_buffer_context * src_ctx = (ggml_backend_rdma_buffer_context *)src->buffer->context;
    ggml_backend_rdma_buffer_context * dst_ctx = (ggml_backend_rdma_buffer_context *)dst->buffer->context;

    if (src_ctx->conn != dst_ctx->conn) {
        return false;
    }

    ggml_backend_rdma_buffer_context * ctx = (ggml_backend_rdma_buffer_context *)buffer->context;
    rdma_msg_copy_tensor_req request;
    request.src = serialize_tensor(src);
    request.dst = serialize_tensor(dst);
    rdma_msg_copy_tensor_rsp response;
    bool status = send_rdma_cmd_with_rsp(ctx->conn.get(), RDMA_CMD_COPY_TENSOR,
                                          &request, sizeof(request), &response, sizeof(response));
    RDMA_STATUS_ASSERT(status);
    return response.result;
}

static void ggml_backend_rdma_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rdma_buffer_context * ctx = (ggml_backend_rdma_buffer_context *)buffer->context;
    rdma_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rdma_cmd(ctx->conn.get(), RDMA_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr);
    RDMA_STATUS_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rdma_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rdma_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rdma_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rdma_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rdma_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rdma_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_rdma_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rdma_buffer_clear,
    /* .reset           = */ NULL,
};

// Buffer type interface

static const char * ggml_backend_rdma_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rdma_buffer_type_context * buft_ctx = (ggml_backend_rdma_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rdma_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rdma_buffer_type_context * buft_ctx = (ggml_backend_rdma_buffer_type_context *)buft->context;
    rdma_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rdma_msg_alloc_buffer_rsp response;

    auto conn = get_connection(buft_ctx->endpoint);
    if (!conn) {
        GGML_LOG_ERROR("[rdma] Failed to connect to %s\n", buft_ctx->endpoint.c_str());
        return nullptr;
    }

    bool status = send_rdma_cmd_with_rsp(conn.get(), RDMA_CMD_ALLOC_BUFFER,
                                          &request, sizeof(request), &response, sizeof(response));
    RDMA_STATUS_ASSERT(status);

    if (response.remote_ptr != 0) {
        auto * ctx = new ggml_backend_rdma_buffer_context {
            conn,
            nullptr,
            response.remote_ptr,
            response.mr_addr,
            response.mr_rkey,
            response.remote_size,
            std::make_unique<rdma_memory_pool>(conn.get())
        };

        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rdma_buffer_interface,
            ctx,
            response.remote_size);
        return buffer;
    }

    return nullptr;
}

static size_t ggml_backend_rdma_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rdma_buffer_type_context * buft_ctx = (ggml_backend_rdma_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t ggml_backend_rdma_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rdma_buffer_type_context * buft_ctx = (ggml_backend_rdma_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

// FNV-1a hash for tensor signature (used for alloc_size cache)
static uint64_t compute_tensor_signature(const ggml_tensor * tensor) {
    constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

    uint64_t hash = FNV_OFFSET;

    auto fnv_hash = [&hash](const void * data, size_t len) {
        const uint8_t * bytes = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < len; i++) {
            hash ^= bytes[i];
            hash *= FNV_PRIME;
        }
    };

    // Hash tensor type
    fnv_hash(&tensor->type, sizeof(tensor->type));

    // Hash dimensions
    fnv_hash(tensor->ne, sizeof(tensor->ne));

    // Hash operation
    fnv_hash(&tensor->op, sizeof(tensor->op));

    // Hash operation parameters
    fnv_hash(tensor->op_params, sizeof(tensor->op_params));

    // Hash source tensor shapes (not pointers)
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            fnv_hash(&tensor->src[i]->type, sizeof(tensor->src[i]->type));
            fnv_hash(tensor->src[i]->ne, sizeof(tensor->src[i]->ne));
        }
    }

    return hash;
}

static size_t ggml_backend_rdma_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_rdma_buffer_type_context * buft_ctx = (ggml_backend_rdma_buffer_type_context *)buft->context;

    // Phase 1: Local calculation for quantized tensors (avoids RPC)
    // This handles the vast majority of tensors during model loading
    if (ggml_is_quantized(tensor->type) &&
        (tensor->ne[0] % RDMA_MATRIX_ROW_PADDING != 0) &&
        (tensor->view_src == nullptr)) {

        size_t size = ggml_nbytes(tensor);
        size += ggml_row_size(tensor->type,
                              RDMA_MATRIX_ROW_PADDING - tensor->ne[0] % RDMA_MATRIX_ROW_PADDING);
        RDMA_LOG_DBG("[RDMA] get_alloc_size local (quantized): %s -> %zu bytes\n", tensor->name, size);
        return size;
    }

    // Phase 2: Cached RPC for FLASH_ATTN_EXT and MUL_MAT_ID operations
    if (tensor->op == GGML_OP_FLASH_ATTN_EXT || tensor->op == GGML_OP_MUL_MAT_ID) {
        uint64_t sig = compute_tensor_signature(tensor);

        // Check cache first
        {
            std::lock_guard<std::mutex> lock(buft_ctx->alloc_size_cache.mutex);
            auto it = buft_ctx->alloc_size_cache.entries.find(sig);
            if (it != buft_ctx->alloc_size_cache.entries.end()) {
                RDMA_LOG_DBG("[RDMA] get_alloc_size cache hit: %s -> %zu bytes\n", tensor->name, it->second);
                return it->second;
            }
        }

        // Cache miss - need RPC
        auto conn = get_connection(buft_ctx->endpoint);
        if (!conn) {
            return ggml_nbytes(tensor);
        }

        rdma_msg_get_alloc_size_req request = {
            buft_ctx->device,
            serialize_tensor(tensor),
            {}
        };

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        rdma_msg_get_alloc_size_rsp response;
        bool status = send_rdma_cmd_with_rsp(conn.get(), RDMA_CMD_GET_ALLOC_SIZE,
                                              &request, sizeof(request), &response, sizeof(response));
        RDMA_STATUS_ASSERT(status);

        // Cache the result
        {
            std::lock_guard<std::mutex> lock(buft_ctx->alloc_size_cache.mutex);
            // Evict oldest entries if cache is full (simple FIFO-like behavior)
            if (buft_ctx->alloc_size_cache.entries.size() >= RDMA_ALLOC_SIZE_CACHE_MAX_ENTRIES) {
                buft_ctx->alloc_size_cache.entries.clear();
            }
            buft_ctx->alloc_size_cache.entries[sig] = response.alloc_size;
        }

        RDMA_LOG_DBG("[RDMA] get_alloc_size RPC: %s -> %zu bytes (cached)\n", tensor->name, response.alloc_size);
        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rdma_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rdma_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rdma_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rdma_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rdma_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rdma_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

// Backend interface

static const char * ggml_backend_rdma_name(ggml_backend_t backend) {
    ggml_backend_rdma_context * ctx = (ggml_backend_rdma_context *)backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_rdma_free(ggml_backend_t backend) {
    ggml_backend_rdma_context * ctx = (ggml_backend_rdma_context *)backend->context;
    delete ctx;
    delete backend;
}

static void ggml_backend_rdma_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // RDMA operations are synchronous in current implementation
}

static void add_tensor(ggml_tensor * tensor, std::vector<rdma_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) return;
    if (visited.find(tensor) != visited.end()) return;

    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(uint32_t device, const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rdma_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;

    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }

    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rdma_tensor);
    output.resize(output_size, 0);

    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);

    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);

    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);

    memcpy(dest, tensors.data(), n_tensors * sizeof(rdma_tensor));
}

static enum ggml_status ggml_backend_rdma_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rdma_context * ctx = (ggml_backend_rdma_context *)backend->context;

    GGML_ASSERT(cgraph->n_nodes > 0);

    bool reuse = ctx->gc.is_cached(cgraph);
    if (reuse) {
        rdma_msg_graph_recompute_req request;
        request.device = ctx->device;
        bool status = send_rdma_cmd(ctx->conn.get(), RDMA_CMD_GRAPH_RECOMPUTE, &request, sizeof(request), nullptr);
        RDMA_STATUS_ASSERT(status);
    } else {
        ctx->gc.add(cgraph);
        std::vector<uint8_t> input;
        serialize_graph(ctx->device, cgraph, input);

        // Register temporary MR for large graph data (> 64KB)
        static const size_t RDMA_SEND_BUF_SIZE = 64 * 1024;
        struct ibv_mr * send_mr = nullptr;
        if (input.size() > RDMA_SEND_BUF_SIZE) {
            send_mr = ctx->conn->register_memory(input.data(), input.size(), IBV_ACCESS_LOCAL_WRITE);
            if (!send_mr) {
                GGML_LOG_ERROR("[rdma] Failed to register MR for graph data (size=%zu)\n", input.size());
                return GGML_STATUS_FAILED;
            }
        }

        bool status = send_rdma_cmd(ctx->conn.get(), RDMA_CMD_GRAPH_COMPUTE, input.data(), input.size(), send_mr);

        if (send_mr) {
            ctx->conn->deregister_memory(send_mr);
        }

        RDMA_STATUS_ASSERT(status);
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_rdma_interface = {
    /* .get_name                = */ ggml_backend_rdma_name,
    /* .free                    = */ ggml_backend_rdma_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rdma_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rdma_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

// Device interface

struct ggml_backend_rdma_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_rdma_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rdma_device_context * ctx = (ggml_backend_rdma_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_rdma_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rdma_device_context * ctx = (ggml_backend_rdma_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_rdma_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rdma_device_context * ctx = (ggml_backend_rdma_device_context *)dev->context;

    auto conn = get_connection(ctx->endpoint);
    if (!conn) {
        *free = 0;
        *total = 0;
        return;
    }

    rdma_msg_get_device_memory_req request;
    request.device = ctx->device;
    rdma_msg_get_device_memory_rsp response;
    bool status = send_rdma_cmd_with_rsp(conn.get(), RDMA_CMD_GET_DEVICE_MEMORY,
                                          &request, sizeof(request), &response, sizeof(response));
    if (status) {
        *free = response.free_mem;
        *total = response.total_mem;
    } else {
        *free = 0;
        *total = 0;
    }
}

static enum ggml_backend_dev_type ggml_backend_rdma_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_rdma_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rdma_device_get_name(dev);
    props->description = ggml_backend_rdma_device_get_description(dev);
    props->type = ggml_backend_rdma_device_get_type(dev);
    ggml_backend_rdma_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_rdma_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_rdma_device_context * dev_ctx = (ggml_backend_rdma_device_context *)dev->context;
    return ggml_backend_rdma_init(dev_ctx->endpoint.c_str(), dev_ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_rdma_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rdma_device_context * ctx = (ggml_backend_rdma_device_context *)dev->context;
    return ggml_backend_rdma_buffer_type(ctx->endpoint.c_str(), ctx->device);
}

static bool ggml_backend_rdma_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return true;  // All operations are supported remotely
}

static bool ggml_backend_rdma_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || !buft->iface.get_name) {
        return false;
    }
    const char * name = buft->iface.get_name(buft);
    if (!name) {
        return false;
    }

    ggml_backend_rdma_device_context * ctx = (ggml_backend_rdma_device_context *)dev->context;
    // Support RDMA buffers from same endpoint
    return strstr(name, ctx->endpoint.c_str()) != nullptr;
}

static ggml_backend_device_i ggml_backend_rdma_device_interface = {
    /* .get_name             = */ ggml_backend_rdma_device_get_name,
    /* .get_description      = */ ggml_backend_rdma_device_get_description,
    /* .get_memory           = */ ggml_backend_rdma_device_get_memory,
    /* .get_type             = */ ggml_backend_rdma_device_get_type,
    /* .get_props            = */ ggml_backend_rdma_device_get_props,
    /* .init_backend         = */ ggml_backend_rdma_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_rdma_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rdma_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rdma_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// Registry interface

struct ggml_backend_rdma_reg_context {
    std::string endpoint;
    std::vector<ggml_backend_dev_t> devices;
    std::string name;
};

static const char * ggml_backend_rdma_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rdma_reg_context * ctx = (ggml_backend_rdma_reg_context *)reg->context;
    return ctx->name.c_str();
}

static size_t ggml_backend_rdma_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rdma_reg_context * ctx = (ggml_backend_rdma_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_rdma_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rdma_reg_context * ctx = (ggml_backend_rdma_reg_context *)reg->context;
    if (index >= ctx->devices.size()) {
        return nullptr;
    }
    return ctx->devices[index];
}

static void * ggml_backend_rdma_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_rdma_start_server") == 0) {
        return (void *)ggml_backend_rdma_start_server;
    }
    if (strcmp(name, "ggml_backend_rdma_enable_gdr") == 0) {
        return (void *)ggml_backend_rdma_enable_gdr;
    }
    if (strcmp(name, "ggml_backend_rdma_gdr_available") == 0) {
        return (void *)ggml_backend_rdma_gdr_available;
    }
    return nullptr;
}

static ggml_backend_reg_i ggml_backend_rdma_reg_interface = {
    /* .get_name         = */ ggml_backend_rdma_reg_get_name,
    /* .get_device_count = */ ggml_backend_rdma_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rdma_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rdma_reg_get_proc_address,
};

// Public API implementations

ggml_backend_t ggml_backend_rdma_init(const char * endpoint, uint32_t device) {
    auto conn = get_connection(endpoint);
    if (!conn) {
        GGML_LOG_ERROR("[rdma] Failed to connect to %s\n", endpoint);
        return nullptr;
    }

    if (!check_server_version(conn.get())) {
        return nullptr;
    }

    std::string dev_name = "RDMA" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rdma_context * ctx = new ggml_backend_rdma_context {
        endpoint,
        device,
        dev_name,
        {},
        conn,
        false,
        {}
    };

    auto reg = ggml_backend_rdma_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        ggml_backend_rdma_guid(),
        ggml_backend_rdma_interface,
        ggml_backend_reg_dev_get(reg, device),
        ctx
    };

    return backend;
}

bool ggml_backend_is_rdma(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rdma_guid());
}

ggml_backend_buffer_type_t ggml_backend_rdma_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    std::string buft_name = "RDMA" + std::to_string(device) + "[" + std::string(endpoint) + "]";

    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }

    auto conn = get_connection(endpoint);
    if (!conn) {
        GGML_LOG_ERROR("[rdma] Failed to connect to %s\n", endpoint);
        return nullptr;
    }

    // Get alignment
    rdma_msg_get_alignment_req align_req = {device};
    rdma_msg_get_alignment_rsp align_rsp;
    if (!send_rdma_cmd_with_rsp(conn.get(), RDMA_CMD_GET_ALIGNMENT,
                                 &align_req, sizeof(align_req), &align_rsp, sizeof(align_rsp))) {
        return nullptr;
    }

    // Get max size
    rdma_msg_get_max_size_req size_req = {device};
    rdma_msg_get_max_size_rsp size_rsp;
    if (!send_rdma_cmd_with_rsp(conn.get(), RDMA_CMD_GET_MAX_SIZE,
                                 &size_req, sizeof(size_req), &size_rsp, sizeof(size_rsp))) {
        return nullptr;
    }

    ggml_backend_rdma_buffer_type_context * buft_ctx = new ggml_backend_rdma_buffer_type_context {
        endpoint,
        device,
        buft_name,
        align_rsp.alignment,
        size_rsp.max_size
    };

    auto reg = ggml_backend_rdma_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        ggml_backend_rdma_buffer_type_interface,
        ggml_backend_reg_dev_get(reg, device),
        buft_ctx
    };

    buft_map[buft_name] = buft;
    return buft;
}

void ggml_backend_rdma_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto conn = get_connection(endpoint);
    if (!conn) {
        *free = 0;
        *total = 0;
        return;
    }

    rdma_msg_get_device_memory_req request;
    request.device = device;
    rdma_msg_get_device_memory_rsp response;
    bool status = send_rdma_cmd_with_rsp(conn.get(), RDMA_CMD_GET_DEVICE_MEMORY,
                                          &request, sizeof(request), &response, sizeof(response));
    if (status) {
        *free = response.free_mem;
        *total = response.total_mem;
    } else {
        *free = 0;
        *total = 0;
    }
}

// Helper function to parse comma-separated server list
static std::vector<std::string> parse_server_list(const char * servers) {
    std::vector<std::string> result;
    if (!servers || servers[0] == '\0') {
        return result;
    }

    std::string s(servers);
    size_t pos = 0;
    while ((pos = s.find(',')) != std::string::npos) {
        std::string server = s.substr(0, pos);
        if (!server.empty()) {
            result.push_back(server);
        }
        s.erase(0, pos + 1);
    }
    if (!s.empty()) {
        result.push_back(s);
    }
    return result;
}

ggml_backend_reg_t ggml_backend_rdma_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;
    static ggml_backend_rdma_reg_context * reg_ctx = nullptr;

    if (!initialized) {
        reg.api_version = GGML_BACKEND_API_VERSION;
        reg.iface = ggml_backend_rdma_reg_interface;
        reg_ctx = new ggml_backend_rdma_reg_context {
            "",
            {},
            "RDMA"
        };
        reg.context = reg_ctx;

        // Check for GGML_RDMA_SERVERS environment variable
        const char * servers_env = std::getenv("GGML_RDMA_SERVERS");
        if (servers_env) {
            auto servers = parse_server_list(servers_env);
            for (const auto & server : servers) {
                GGML_LOG_INFO("[rdma] Adding server from environment: %s\n", server.c_str());
                auto conn = get_connection(server);
                if (!conn) {
                    GGML_LOG_ERROR("[rdma] Failed to connect to %s\n", server.c_str());
                    continue;
                }

                // Get device count
                rdma_msg_device_count_rsp dev_count_rsp;
                if (!send_rdma_cmd_with_rsp(conn.get(), RDMA_CMD_DEVICE_COUNT, nullptr, 0,
                                             &dev_count_rsp, sizeof(dev_count_rsp))) {
                    GGML_LOG_ERROR("[rdma] Failed to get device count from %s\n", server.c_str());
                    continue;
                }

                // Add devices to registry
                for (uint32_t i = 0; i < dev_count_rsp.device_count; i++) {
                    auto * dev_ctx = new ggml_backend_rdma_device_context;
                    dev_ctx->endpoint = server;
                    dev_ctx->device = i;
                    dev_ctx->name = "RDMA" + std::to_string(i) + "[" + server + "]";
                    dev_ctx->description = "Remote RDMA device " + std::to_string(i) + " on " + server;

                    auto * dev = new ggml_backend_device;
                    dev->iface = ggml_backend_rdma_device_interface;
                    dev->reg = &reg;
                    dev->context = dev_ctx;

                    reg_ctx->devices.push_back(dev);
                    GGML_LOG_INFO("[rdma] Added device: %s\n", dev_ctx->name.c_str());
                }
            }
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_reg_t ggml_backend_rdma_add_server(const char * endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    auto it = reg_map.find(endpoint);
    if (it != reg_map.end()) {
        return it->second;
    }

    auto conn = get_connection(endpoint);
    if (!conn) {
        return nullptr;
    }

    // Get device count
    rdma_msg_device_count_rsp dev_count_rsp;
    if (!send_rdma_cmd_with_rsp(conn.get(), RDMA_CMD_DEVICE_COUNT, nullptr, 0,
                                 &dev_count_rsp, sizeof(dev_count_rsp))) {
        return nullptr;
    }

    // Create registry context
    auto * reg_ctx = new ggml_backend_rdma_reg_context;
    reg_ctx->endpoint = endpoint;
    reg_ctx->name = "RDMA[" + std::string(endpoint) + "]";

    // Create registry
    auto * reg = new ggml_backend_reg;
    reg->api_version = GGML_BACKEND_API_VERSION;
    reg->iface = ggml_backend_rdma_reg_interface;
    reg->context = reg_ctx;

    // Create devices
    for (uint32_t i = 0; i < dev_count_rsp.device_count; i++) {
        auto * dev_ctx = new ggml_backend_rdma_device_context;
        dev_ctx->endpoint = endpoint;
        dev_ctx->device = i;
        dev_ctx->name = "RDMA" + std::to_string(i) + "[" + std::string(endpoint) + "]";
        dev_ctx->description = "Remote RDMA device " + std::to_string(i);

        auto * dev = new ggml_backend_device;
        dev->iface = ggml_backend_rdma_device_interface;
        dev->reg = reg;
        dev->context = dev_ctx;

        reg_ctx->devices.push_back(dev);
    }

    reg_map[endpoint] = reg;
    return reg;
}

bool ggml_backend_rdma_enable_gdr(ggml_backend_t backend) {
    if (!ggml_backend_is_rdma(backend)) {
        return false;
    }

    if (!ggml_backend_rdma_gdr_available()) {
        return false;
    }

    ggml_backend_rdma_context * ctx = (ggml_backend_rdma_context *)backend->context;
    ctx->gdr_enabled = true;
    GGML_LOG_INFO("[rdma] GPUDirect RDMA enabled for backend %s\n", ctx->name.c_str());
    return true;
}

bool ggml_backend_rdma_gdr_available(void) {
    return gdr_memory_manager::is_available();
}

void ggml_backend_rdma_get_stats(ggml_backend_t backend, ggml_rdma_stats_t * stats) {
    if (!ggml_backend_is_rdma(backend) || !stats) {
        return;
    }

    ggml_backend_rdma_context * ctx = (ggml_backend_rdma_context *)backend->context;
    if (ctx->conn) {
        const rdma_stats & s = ctx->conn->get_stats();
        stats->bytes_sent = s.bytes_sent;
        stats->bytes_received = s.bytes_received;
        stats->rdma_writes = s.rdma_writes;
        stats->rdma_reads = s.rdma_reads;
        stats->send_ops = s.send_ops;
        stats->recv_ops = s.recv_ops;
    }
}

void ggml_backend_rdma_reset_stats(ggml_backend_t backend) {
    if (!ggml_backend_is_rdma(backend)) {
        return;
    }

    ggml_backend_rdma_context * ctx = (ggml_backend_rdma_context *)backend->context;
    if (ctx->conn) {
        ctx->conn->reset_stats();
    }
}

// Server implementation

class rdma_server {
public:
    rdma_server(std::vector<ggml_backend_t> backends, const char * cache_dir)
        : backends_(std::move(backends)), cache_dir_(cache_dir) {
        stored_graphs_.resize(backends_.size());
    }

    ~rdma_server() {
        for (auto buffer : buffers_) {
            ggml_backend_buffer_free(buffer);
        }
    }

    void hello(rdma_msg_hello_rsp & response) {
        response.major = RDMA_PROTO_MAJOR_VERSION;
        response.minor = RDMA_PROTO_MINOR_VERSION;
        response.patch = RDMA_PROTO_PATCH_VERSION;
        response.gdr_available = gdr_memory_manager::is_available() ? 1 : 0;
    }

    bool alloc_buffer(const rdma_msg_alloc_buffer_req & request, rdma_msg_alloc_buffer_rsp & response,
                      rdma_connection * conn) {
        uint32_t dev_id = request.device;
        if (dev_id >= backends_.size()) return false;

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends_[dev_id]);
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);

        response.remote_ptr = 0;
        response.remote_size = 0;
        response.mr_addr = 0;
        response.mr_rkey = 0;

        if (buffer != nullptr) {
            response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
            response.remote_size = buffer->size;
            buffers_.insert(buffer);

            // Register buffer for RDMA access
            void * base = ggml_backend_buffer_get_base(buffer);
            int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
            struct ibv_mr * mr = conn->register_memory(base, buffer->size, access_flags);
            if (mr) {
                response.mr_addr = (uint64_t)mr->addr;
                response.mr_rkey = mr->rkey;
                buffer_mrs_[buffer] = mr;
            }

            RDMA_LOG_DBG("[server] Allocated buffer: ptr=%p, size=%zu, mr_addr=0x%lx, mr_rkey=0x%x\n",
                         (void*)response.remote_ptr, (size_t)response.remote_size,
                         response.mr_addr, response.mr_rkey);
        }
        return true;
    }

    bool free_buffer(const rdma_msg_free_buffer_req & request, rdma_connection * conn) {
        ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
        if (buffers_.find(buffer) == buffers_.end()) return false;

        // Deregister MR
        auto mr_it = buffer_mrs_.find(buffer);
        if (mr_it != buffer_mrs_.end()) {
            conn->deregister_memory(mr_it->second);
            buffer_mrs_.erase(mr_it);
        }

        ggml_backend_buffer_free(buffer);
        buffers_.erase(buffer);
        return true;
    }

    bool get_alignment(const rdma_msg_get_alignment_req & request, rdma_msg_get_alignment_rsp & response) {
        uint32_t dev_id = request.device;
        if (dev_id >= backends_.size()) return false;
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends_[dev_id]);
        response.alignment = ggml_backend_buft_get_alignment(buft);
        return true;
    }

    bool get_max_size(const rdma_msg_get_max_size_req & request, rdma_msg_get_max_size_rsp & response) {
        uint32_t dev_id = request.device;
        if (dev_id >= backends_.size()) return false;
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends_[dev_id]);
        response.max_size = ggml_backend_buft_get_max_size(buft);
        return true;
    }

    bool buffer_get_base(const rdma_msg_buffer_get_base_req & request, rdma_msg_buffer_get_base_rsp & response) {
        ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
        if (buffers_.find(buffer) == buffers_.end()) return false;
        void * base = ggml_backend_buffer_get_base(buffer);
        response.base_ptr = reinterpret_cast<uint64_t>(base);
        return true;
    }

    bool buffer_clear(const rdma_msg_buffer_clear_req & request) {
        ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
        if (buffers_.find(buffer) == buffers_.end()) return false;
        ggml_backend_buffer_clear(buffer, request.value);
        return true;
    }

    bool get_device_memory(const rdma_msg_get_device_memory_req & request, rdma_msg_get_device_memory_rsp & response) {
        uint32_t dev_id = request.device;
        if (dev_id >= backends_.size()) return false;
        ggml_backend_dev_t dev = ggml_backend_get_device(backends_[dev_id]);
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        response.free_mem = free;
        response.total_mem = total;
        return true;
    }

    bool set_tensor(const std::vector<uint8_t> & input) {
        // serialization format: | rdma_tensor | offset (8 bytes) | data (size bytes) |
        if (input.size() < sizeof(rdma_tensor) + sizeof(uint64_t)) {
            return false;
        }
        const rdma_tensor * in_tensor = (const rdma_tensor *)input.data();
        uint64_t offset;
        memcpy(&offset, input.data() + sizeof(rdma_tensor), sizeof(offset));
        const size_t size = input.size() - sizeof(rdma_tensor) - sizeof(offset);

        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr { ggml_init(params) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx = ctx_ptr.get();
        ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
        if (tensor == nullptr || tensor->buffer == nullptr) {
            GGML_LOG_ERROR("[rdma_server] error deserializing tensor in set_tensor\n");
            return false;
        }
        RDMA_LOG_DBG("[rdma_server] set_tensor: buffer=%p, data=%p, offset=%" PRIu64 ", size=%zu\n",
                     (void*)tensor->buffer, tensor->data, offset, size);

        // Validate data region
        const size_t p0 = (size_t)ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 ||
            size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[rdma_server] set_tensor: out of bounds\n");
            return false;
        }

        const void * data = input.data() + sizeof(rdma_tensor) + sizeof(offset);
        ggml_backend_tensor_set(tensor, data, offset, size);
        return true;
    }

    bool get_tensor(const rdma_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr { ggml_init(params) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx = ctx_ptr.get();
        ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
        if (tensor == nullptr || tensor->buffer == nullptr) {
            GGML_LOG_ERROR("[rdma_server] error deserializing tensor in get_tensor\n");
            return false;
        }
        RDMA_LOG_DBG("[rdma_server] get_tensor: buffer=%p, data=%p, offset=%" PRIu64 ", size=%" PRIu64 "\n",
                     (void*)tensor->buffer, tensor->data, request.offset, request.size);

        // Validate data region
        const size_t p0 = (size_t)ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[rdma_server] get_tensor: out of bounds\n");
            return false;
        }

        response.resize(request.size, 0);
        ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
        return true;
    }

    bool copy_tensor(const rdma_msg_copy_tensor_req & request, rdma_msg_copy_tensor_rsp & response) {
        struct ggml_init_params params {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr { ggml_init(params) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx = ctx_ptr.get();

        ggml_tensor * src = deserialize_tensor(ctx, &request.src);
        ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
        if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
            GGML_LOG_ERROR("[rdma_server] error deserializing tensors in copy_tensor\n");
            return false;
        }

        RDMA_LOG_DBG("[rdma_server] copy_tensor: src->buffer=%p, dst->buffer=%p\n",
                     (void*)src->buffer, (void*)dst->buffer);

        response.result = ggml_backend_buffer_copy_tensor(src, dst);
        return true;
    }

    bool init_tensor(const rdma_msg_init_tensor_req & request) {
        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr { ggml_init(params) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx = ctx_ptr.get();
        ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
        if (tensor == nullptr) {
            GGML_LOG_ERROR("[rdma_server] null tensor in init_tensor\n");
            return false;
        }
        RDMA_LOG_DBG("[rdma_server] init_tensor: buffer=%p, data=%p\n",
                     (void*)tensor->buffer, tensor->data);

        ggml_backend_buffer_t buffer = tensor->buffer;
        if (buffer && buffer->iface.init_tensor) {
            buffer->iface.init_tensor(buffer, tensor);
        }
        return true;
    }

    bool get_alloc_size(const rdma_msg_get_alloc_size_req & request, rdma_msg_get_alloc_size_rsp & response) {
        uint32_t dev_id = request.device;
        if (dev_id >= backends_.size()) return false;

        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * (1 + GGML_MAX_SRC),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr { ggml_init(params) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx = ctx_ptr.get();

        ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
        if (tensor == nullptr) {
            GGML_LOG_ERROR("[rdma_server] null tensor in get_alloc_size\n");
            return false;
        }
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (request.srcs[i].id != 0) {
                tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
            }
        }

        ggml_backend_buffer_type_t buft;
        if (tensor->buffer == nullptr) {
            buft = ggml_backend_get_default_buffer_type(backends_[dev_id]);
        } else {
            buft = tensor->buffer->buft;
        }

        response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
        return true;
    }

    bool graph_compute(const std::vector<uint8_t> & input) {
        // serialization format:
        // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rdma_tensor)) |
        if (input.size() < 2*sizeof(uint32_t)) {
            return false;
        }
        const uint8_t * src = input.data();
        uint32_t device;
        memcpy(&device, src, sizeof(device));
        src += sizeof(device);
        if (device >= backends_.size()) {
            return false;
        }
        uint32_t n_nodes;
        memcpy(&n_nodes, src, sizeof(n_nodes));
        src += sizeof(n_nodes);
        if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
            return false;
        }
        const uint64_t * nodes = (const uint64_t *)src;
        src += n_nodes*sizeof(uint64_t);
        uint32_t n_tensors;
        memcpy(&n_tensors, src, sizeof(n_tensors));
        src += sizeof(n_tensors);
        if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rdma_tensor)) {
            return false;
        }
        const rdma_tensor * tensors = (const rdma_tensor *)src;
        RDMA_LOG_DBG("[rdma_server] graph_compute: device=%u, n_nodes=%u, n_tensors=%u\n", device, n_nodes, n_tensors);

        size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);

        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr { ggml_init(params) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx = ctx_ptr.get();
        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
        graph->n_nodes = n_nodes;
        std::unordered_map<uint64_t, const rdma_tensor*> tensor_ptrs;
        tensor_ptrs.reserve(n_tensors);
        for (uint32_t i = 0; i < n_tensors; i++) {
            tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
        }
        std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
        tensor_map.reserve(n_nodes);
        for (uint32_t i = 0; i < n_nodes; i++) {
            int64_t id;
            memcpy(&id, &nodes[i], sizeof(id));
            graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
            if (graph->nodes[i] == nullptr && id != 0) {
                GGML_LOG_ERROR("[rdma_server] failed to create graph node %u\n", i);
                return false;
            }
        }
        ggml_status status = ggml_backend_graph_compute(backends_[device], graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
        stored_graphs_[device].ctx_ptr.swap(ctx_ptr);
        stored_graphs_[device].graph = graph;
        return true;
    }

    bool graph_recompute(const rdma_msg_graph_recompute_req & request) {
        uint32_t device = request.device;
        if (device >= backends_.size()) {
            return false;
        }
        if (stored_graphs_[device].graph == nullptr) {
            return false;
        }
        ggml_cgraph * graph = stored_graphs_[device].graph;
        RDMA_LOG_DBG("[rdma_server] graph_recompute: device=%u\n", device);
        ggml_status status = ggml_backend_graph_compute(backends_[device], graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
        return true;
    }

private:
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rdma_tensor * tensor) {
        if (tensor->type >= GGML_TYPE_COUNT) {
            GGML_LOG_ERROR("[rdma_server] invalid tensor type: %u\n", tensor->type);
            return nullptr;
        }

        ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type)tensor->type,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        if (result == nullptr) {
            return nullptr;
        }

        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            result->nb[i] = tensor->nb[i];
        }
        result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
        if (result->buffer && buffers_.find(result->buffer) == buffers_.end()) {
            result->buffer = nullptr;
        }

        if (result->buffer) {
            uint64_t tensor_size = (uint64_t)ggml_nbytes(result);
            uint64_t buffer_start = (uint64_t)ggml_backend_buffer_get_base(result->buffer);
            uint64_t buffer_size = (uint64_t)ggml_backend_buffer_get_size(result->buffer);
            GGML_ASSERT(tensor->data + tensor_size >= tensor->data);
            GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
        }

        result->op = (ggml_op)tensor->op;
        for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
            result->op_params[i] = tensor->op_params[i];
        }
        result->flags = tensor->flags;
        result->data = reinterpret_cast<void *>(tensor->data);
        ggml_set_name(result, tensor->name);
        return result;
    }

    ggml_tensor * create_node(uint64_t id, struct ggml_context * ctx,
                               const std::unordered_map<uint64_t, const rdma_tensor*> & tensor_ptrs,
                               std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
        if (tensor_map.find(id) != tensor_map.end()) {
            return tensor_map[id];
        }
        auto it_ptr = tensor_ptrs.find(id);
        if (it_ptr == tensor_ptrs.end()) {
            return nullptr;
        }
        const rdma_tensor * tensor = it_ptr->second;

        struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
        if (result == nullptr) {
            return nullptr;
        }
        tensor_map[id] = result;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == 0) {
                result->src[i] = nullptr;
            } else {
                result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
                if (result->src[i] == nullptr) {
                    return nullptr;
                }
            }
        }
        if (tensor->view_src == 0) {
            result->view_src = nullptr;
        } else {
            result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
            if (result->view_src == nullptr) {
                return nullptr;
            }
        }
        result->view_offs = tensor->view_offs;
        return result;
    }

    std::vector<ggml_backend_t> backends_;
    const char * cache_dir_;
    std::unordered_set<ggml_backend_buffer_t> buffers_;
    std::unordered_map<ggml_backend_buffer_t, struct ibv_mr *> buffer_mrs_;

    struct stored_graph {
        ggml_context_ptr ctx_ptr;
        ggml_cgraph * graph;
    };
    std::vector<stored_graph> stored_graphs_;
};

void ggml_backend_rdma_start_server(const char * endpoint, const char * cache_dir,
                                     size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rdma_start_server\n");
        return;
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        fprintf(stderr, "Invalid endpoint format: %s\n", endpoint);
        return;
    }

    // Initialize backends
    std::vector<ggml_backend_t> backends;
    printf("Starting RDMA server v%d.%d.%d\n",
           RDMA_PROTO_MAJOR_VERSION, RDMA_PROTO_MINOR_VERSION, RDMA_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("  GPUDirect RDMA : %s\n", gdr_memory_manager::is_available() ? "available" : "not available");
    printf("Devices:\n");

    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n",
               ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);

        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", ggml_backend_dev_name(dev));
            return;
        }
        backends.push_back(backend);

        // Set thread count if supported
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg) {
            auto set_threads_fn = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (set_threads_fn) {
                set_threads_fn(backend, n_threads);
            }
        }
    }

    // Start RDMA server
    rdma_connection_manager & manager = get_connection_manager();
    if (!manager.start_server(host.c_str(), port)) {
        fprintf(stderr, "Failed to start RDMA server\n");
        return;
    }

    printf("RDMA server listening on %s:%d\n", host.c_str(), port);

    // Accept connections
    while (true) {
        auto conn = manager.accept_connection();
        if (!conn) {
            fprintf(stderr, "Failed to accept connection\n");
            continue;
        }

        printf("Accepted RDMA connection from %s\n", conn->get_endpoint().c_str());

        // Handle client in current thread (single-threaded for simplicity)
        // In production, this should be multi-threaded
        rdma_server server(backends, cache_dir);

        // Process commands
        uint8_t cmd;
        while (conn->recv(&cmd, 1, nullptr)) {
            if (cmd >= RDMA_CMD_COUNT) {
                fprintf(stderr, "Unknown command: %d\n", cmd);
                break;
            }

            // Read message size
            uint64_t msg_size;
            if (!conn->recv(&msg_size, sizeof(msg_size), nullptr)) {
                break;
            }

            // Read message data
            std::vector<uint8_t> msg_data(msg_size);
            if (msg_size > 0) {
                static const size_t RDMA_RECV_BUF_SIZE = 64 * 1024;
                struct ibv_mr * recv_mr = nullptr;
                if (msg_size > RDMA_RECV_BUF_SIZE) {
                    recv_mr = conn->register_memory(msg_data.data(), msg_size, IBV_ACCESS_LOCAL_WRITE);
                    if (!recv_mr) {
                        fprintf(stderr, "[rdma-server] Failed to register MR for recv (size=%zu)\n", (size_t)msg_size);
                        break;
                    }
                }
                bool ok = conn->recv(msg_data.data(), msg_size, recv_mr);
                if (recv_mr) {
                    conn->deregister_memory(recv_mr);
                }
                if (!ok) {
                    break;
                }
            }

            switch (cmd) {
                case RDMA_CMD_HELLO: {
                    rdma_msg_hello_rsp response;
                    server.hello(response);
                    uint64_t rsp_size = sizeof(response);
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    conn->send(&response, sizeof(response), nullptr);
                    break;
                }
                case RDMA_CMD_DEVICE_COUNT: {
                    rdma_msg_device_count_rsp response;
                    response.device_count = backends.size();
                    uint64_t rsp_size = sizeof(response);
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    conn->send(&response, sizeof(response), nullptr);
                    break;
                }
                case RDMA_CMD_ALLOC_BUFFER: {
                    rdma_msg_alloc_buffer_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    rdma_msg_alloc_buffer_rsp response;
                    server.alloc_buffer(request, response, conn.get());
                    uint64_t rsp_size = sizeof(response);
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    conn->send(&response, sizeof(response), nullptr);
                    break;
                }
                case RDMA_CMD_FREE_BUFFER: {
                    rdma_msg_free_buffer_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    server.free_buffer(request, conn.get());
                    uint64_t rsp_size = 0;
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    break;
                }
                case RDMA_CMD_GET_ALIGNMENT: {
                    rdma_msg_get_alignment_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    rdma_msg_get_alignment_rsp response;
                    server.get_alignment(request, response);
                    uint64_t rsp_size = sizeof(response);
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    conn->send(&response, sizeof(response), nullptr);
                    break;
                }
                case RDMA_CMD_GET_MAX_SIZE: {
                    rdma_msg_get_max_size_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    rdma_msg_get_max_size_rsp response;
                    server.get_max_size(request, response);
                    uint64_t rsp_size = sizeof(response);
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    conn->send(&response, sizeof(response), nullptr);
                    break;
                }
                case RDMA_CMD_BUFFER_GET_BASE: {
                    rdma_msg_buffer_get_base_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    rdma_msg_buffer_get_base_rsp response;
                    server.buffer_get_base(request, response);
                    uint64_t rsp_size = sizeof(response);
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    conn->send(&response, sizeof(response), nullptr);
                    break;
                }
                case RDMA_CMD_BUFFER_CLEAR: {
                    rdma_msg_buffer_clear_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    server.buffer_clear(request);
                    uint64_t rsp_size = 0;
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    break;
                }
                case RDMA_CMD_GET_DEVICE_MEMORY: {
                    rdma_msg_get_device_memory_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    rdma_msg_get_device_memory_rsp response;
                    server.get_device_memory(request, response);
                    uint64_t rsp_size = sizeof(response);
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    conn->send(&response, sizeof(response), nullptr);
                    break;
                }
                case RDMA_CMD_SET_TENSOR: {
                    if (!server.set_tensor(msg_data)) {
                        fprintf(stderr, "set_tensor failed\n");
                    }
                    uint64_t rsp_size = 0;
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    break;
                }
                case RDMA_CMD_GET_TENSOR: {
                    rdma_msg_get_tensor_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    std::vector<uint8_t> response;
                    if (!server.get_tensor(request, response)) {
                        fprintf(stderr, "get_tensor failed\n");
                        uint64_t rsp_size = 0;
                        conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    } else {
                        uint64_t rsp_size = response.size();
                        conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                        if (rsp_size > 0) {
                            conn->send(response.data(), response.size(), nullptr);
                        }
                    }
                    break;
                }
                case RDMA_CMD_COPY_TENSOR: {
                    rdma_msg_copy_tensor_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    rdma_msg_copy_tensor_rsp response;
                    if (!server.copy_tensor(request, response)) {
                        fprintf(stderr, "copy_tensor failed\n");
                        response.result = 0;
                    }
                    uint64_t rsp_size = sizeof(response);
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    conn->send(&response, sizeof(response), nullptr);
                    break;
                }
                case RDMA_CMD_INIT_TENSOR: {
                    rdma_msg_init_tensor_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    if (!server.init_tensor(request)) {
                        fprintf(stderr, "init_tensor failed\n");
                    }
                    uint64_t rsp_size = 0;
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    break;
                }
                case RDMA_CMD_GET_ALLOC_SIZE: {
                    rdma_msg_get_alloc_size_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    rdma_msg_get_alloc_size_rsp response;
                    if (!server.get_alloc_size(request, response)) {
                        fprintf(stderr, "get_alloc_size failed\n");
                        response.alloc_size = 0;
                    }
                    uint64_t rsp_size = sizeof(response);
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    conn->send(&response, sizeof(response), nullptr);
                    break;
                }
                case RDMA_CMD_GRAPH_COMPUTE: {
                    if (!server.graph_compute(msg_data)) {
                        fprintf(stderr, "graph_compute failed\n");
                    }
                    uint64_t rsp_size = 0;
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    break;
                }
                case RDMA_CMD_GRAPH_RECOMPUTE: {
                    rdma_msg_graph_recompute_req request;
                    memcpy(&request, msg_data.data(), sizeof(request));
                    if (!server.graph_recompute(request)) {
                        fprintf(stderr, "graph_recompute failed\n");
                    }
                    uint64_t rsp_size = 0;
                    conn->send(&rsp_size, sizeof(rsp_size), nullptr);
                    break;
                }
                default:
                    fprintf(stderr, "Unhandled command: %d\n", cmd);
                    break;
            }
        }

        printf("Client disconnected: %s\n", conn->get_endpoint().c_str());
    }

    // Cleanup
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

// Dynamic loading support
GGML_BACKEND_DL_IMPL(ggml_backend_rdma_reg)
