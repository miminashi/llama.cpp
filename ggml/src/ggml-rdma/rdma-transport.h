#pragma once

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <string>
#include <atomic>

namespace ggml_rdma {

// Forward declarations
class rdma_memory_region;
class rdma_memory_pool;

// RDMA connection configuration
struct rdma_config {
    uint32_t max_send_wr    = 128;      // Maximum send work requests
    uint32_t max_recv_wr    = 128;      // Maximum receive work requests
    uint32_t max_send_sge   = 1;        // Maximum scatter/gather elements per send
    uint32_t max_recv_sge   = 1;        // Maximum scatter/gather elements per receive
    uint32_t max_inline     = 256;      // Maximum inline data size
    uint32_t cq_size        = 256;      // Completion queue size
    uint32_t timeout        = 14;       // Connection timeout (4.096us * 2^timeout)
    uint32_t retry_count    = 7;        // Number of retries
    uint32_t rnr_retry      = 7;        // Receiver not ready retry
    size_t   recv_buf_size  = 16 * 1024 * 1024; // Receive buffer size for control messages (16MB)
};

// Remote memory region information for RDMA operations
struct remote_memory_info {
    uint64_t addr;      // Remote virtual address
    uint32_t rkey;      // Remote key for RDMA access
    size_t   size;      // Size of the region
};

// RDMA transport statistics
struct rdma_stats {
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> rdma_writes{0};
    std::atomic<uint64_t> rdma_reads{0};
    std::atomic<uint64_t> send_ops{0};
    std::atomic<uint64_t> recv_ops{0};
};

// Work request context for tracking completions
struct wr_context {
    enum type { SEND, RECV, RDMA_WRITE, RDMA_READ } op_type;
    void * user_data;
    size_t size;
};

// RDMA connection class - manages a single RDMA connection
class rdma_connection {
public:
    rdma_connection();
    ~rdma_connection();

    // Disable copy
    rdma_connection(const rdma_connection&) = delete;
    rdma_connection& operator=(const rdma_connection&) = delete;

    // Connection establishment
    bool connect(const char * host, int port, const rdma_config & config = {});
    bool accept(struct rdma_cm_id * cm_id, const rdma_config & config = {});
    void disconnect();
    bool is_connected() const { return connected_; }

    // Memory registration
    struct ibv_mr * register_memory(void * addr, size_t size, int access_flags);
    void deregister_memory(struct ibv_mr * mr);

    // RDMA Send/Receive (for control messages)
    bool send(const void * data, size_t size, struct ibv_mr * mr = nullptr);
    bool recv(void * data, size_t size, struct ibv_mr * mr = nullptr, int timeout_ms = -1);

    // Post receive buffer (for async receive)
    bool post_recv(void * data, size_t size, struct ibv_mr * mr, uint64_t wr_id = 0);

    // RDMA Write (one-sided, writes to remote memory)
    bool rdma_write(const void * local_data, size_t size, struct ibv_mr * local_mr,
                    const remote_memory_info & remote, bool signaled = true);

    // RDMA Read (one-sided, reads from remote memory)
    bool rdma_read(void * local_data, size_t size, struct ibv_mr * local_mr,
                   const remote_memory_info & remote, bool signaled = true);

    // Poll completion queue
    // Returns number of completions, or -1 on error
    int poll_cq(int max_entries = 1);

    // Wait for completion (blocking)
    bool wait_for_completion(int timeout_ms = -1);

    // Get local memory region info for sharing with remote peer
    remote_memory_info get_local_mr_info(struct ibv_mr * mr) const;

    // Get connection identifier
    const std::string & get_endpoint() const { return endpoint_; }

    // Get statistics
    const rdma_stats & get_stats() const { return stats_; }
    void reset_stats();

    // Get protection domain (for memory pool)
    struct ibv_pd * get_pd() const { return pd_; }

    // Get RDMA CM ID (for server)
    struct rdma_cm_id * get_cm_id() const { return cm_id_; }

private:
    bool setup_qp(const rdma_config & config);
    bool connect_qp();

    struct rdma_cm_id *     cm_id_          = nullptr;
    struct rdma_event_channel * event_channel_ = nullptr;
    struct ibv_pd *         pd_             = nullptr;
    struct ibv_cq *         cq_             = nullptr;
    struct ibv_qp *         qp_             = nullptr;
    struct ibv_comp_channel * comp_channel_ = nullptr;

    // Internal receive buffer for control messages
    std::vector<uint8_t>    recv_buffer_;
    struct ibv_mr *         recv_mr_        = nullptr;

    // Internal send buffer for control messages (when no external MR provided)
    std::vector<uint8_t>    send_buffer_;
    struct ibv_mr *         send_mr_        = nullptr;

    bool                    connected_      = false;
    std::string             endpoint_;
    rdma_config             config_;
    rdma_stats              stats_;
    std::mutex              send_mutex_;

public:
    // Operation mutex: protects send+recv command sequences from interleaving
    // Must be held by callers doing multi-step operations (send header + send data + recv response)
    // Uses recursive_mutex so that compound operations (e.g., RDMA Write + FLUSH_STAGING)
    // can call send_rdma_cmd while already holding the lock.
    std::recursive_mutex    op_mutex_;
    std::mutex              poll_mutex_;
    std::atomic<bool>       compute_pending_{false};
};

// RDMA connection manager - manages multiple connections and server socket
class rdma_connection_manager {
public:
    rdma_connection_manager();
    ~rdma_connection_manager();

    // Disable copy
    rdma_connection_manager(const rdma_connection_manager&) = delete;
    rdma_connection_manager& operator=(const rdma_connection_manager&) = delete;

    // Get or create connection to endpoint (thread-safe)
    std::shared_ptr<rdma_connection> get_connection(const std::string & endpoint);

    // Server operations
    bool start_server(const char * host, int port, const rdma_config & config = {});
    void stop_server();

    // Accept new connection (blocking)
    std::shared_ptr<rdma_connection> accept_connection();

    // Check if server is running
    bool is_server_running() const { return server_running_; }

private:
    std::mutex connections_mutex_;
    std::unordered_map<std::string, std::shared_ptr<rdma_connection>> connections_;

    struct rdma_cm_id *     server_id_      = nullptr;
    struct rdma_event_channel * server_channel_ = nullptr;
    rdma_config             server_config_;
    bool                    server_running_ = false;
};

// Parse endpoint string "host:port"
bool parse_endpoint(const std::string & endpoint, std::string & host, int & port);

} // namespace ggml_rdma
