#include "rdma-transport.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <chrono>
#include <cstring>
#include <cerrno>
#include <netdb.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>

static const char * RDMA_DEBUG = std::getenv("GGML_RDMA_DEBUG");
static const bool RDMA_NO_SELECTIVE_SIGNAL = !!std::getenv("GGML_RDMA_NO_SELECTIVE_SIGNAL");

static int get_rdma_timeout_ms() {
    const char * env = std::getenv("GGML_RDMA_TIMEOUT_MS");
    if (env) {
        int val = std::atoi(env);
        if (val > 0) return val;
    }
    return 30000;
}
static const int RDMA_TIMEOUT_MS = get_rdma_timeout_ms();

#define RDMA_LOG_DBG(...) \
    do { if (RDMA_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

namespace ggml_rdma {

bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return port > 0 && port <= 65535;
}

// rdma_connection implementation

rdma_connection::rdma_connection() {
}

rdma_connection::~rdma_connection() {
    disconnect();
}

bool rdma_connection::connect(const char * host, int port, const rdma_config & config) {
    config_ = config;
    endpoint_ = std::string(host) + ":" + std::to_string(port);

    // Create event channel
    event_channel_ = rdma_create_event_channel();
    if (!event_channel_) {
        GGML_LOG_ERROR("[rdma_connection] Failed to create event channel: %s\n", strerror(errno));
        return false;
    }

    // Create CM ID
    if (rdma_create_id(event_channel_, &cm_id_, nullptr, RDMA_PS_TCP) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to create CM ID: %s\n", strerror(errno));
        rdma_destroy_event_channel(event_channel_);
        event_channel_ = nullptr;
        return false;
    }

    // Resolve address
    struct addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo * res = nullptr;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    if (getaddrinfo(host, port_str, &hints, &res) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to resolve address %s:%d\n", host, port);
        rdma_destroy_id(cm_id_);
        rdma_destroy_event_channel(event_channel_);
        cm_id_ = nullptr;
        event_channel_ = nullptr;
        return false;
    }

    if (rdma_resolve_addr(cm_id_, nullptr, res->ai_addr, 2000) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to resolve RDMA address: %s\n", strerror(errno));
        freeaddrinfo(res);
        rdma_destroy_id(cm_id_);
        rdma_destroy_event_channel(event_channel_);
        cm_id_ = nullptr;
        event_channel_ = nullptr;
        return false;
    }
    freeaddrinfo(res);

    // Wait for address resolved event
    struct rdma_cm_event * event = nullptr;
    if (rdma_get_cm_event(event_channel_, &event) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to get CM event: %s\n", strerror(errno));
        goto cleanup;
    }
    if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
        GGML_LOG_ERROR("[rdma_connection] Unexpected event: %d (expected ADDR_RESOLVED)\n", event->event);
        rdma_ack_cm_event(event);
        goto cleanup;
    }
    rdma_ack_cm_event(event);

    // Resolve route
    if (rdma_resolve_route(cm_id_, 2000) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to resolve route: %s\n", strerror(errno));
        goto cleanup;
    }

    // Wait for route resolved event
    if (rdma_get_cm_event(event_channel_, &event) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to get CM event: %s\n", strerror(errno));
        goto cleanup;
    }
    if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
        GGML_LOG_ERROR("[rdma_connection] Unexpected event: %d (expected ROUTE_RESOLVED)\n", event->event);
        rdma_ack_cm_event(event);
        goto cleanup;
    }
    rdma_ack_cm_event(event);

    // Setup QP
    if (!setup_qp(config)) {
        goto cleanup;
    }

    // Connect
    if (!connect_qp()) {
        goto cleanup;
    }

    connected_ = true;
    RDMA_LOG_DBG("[rdma_connection] Connected to %s\n", endpoint_.c_str());
    return true;

cleanup:
    if (qp_) {
        ibv_destroy_qp(qp_);
        qp_ = nullptr;
    }
    if (cq_) {
        ibv_destroy_cq(cq_);
        cq_ = nullptr;
    }
    if (comp_channel_) {
        ibv_destroy_comp_channel(comp_channel_);
        comp_channel_ = nullptr;
    }
    if (pd_) {
        ibv_dealloc_pd(pd_);
        pd_ = nullptr;
    }
    if (cm_id_) {
        rdma_destroy_id(cm_id_);
        cm_id_ = nullptr;
    }
    if (event_channel_) {
        rdma_destroy_event_channel(event_channel_);
        event_channel_ = nullptr;
    }
    return false;
}

bool rdma_connection::accept(struct rdma_cm_id * cm_id, const rdma_config & config) {
    config_ = config;
    cm_id_ = cm_id;

    // Get endpoint info
    struct sockaddr_in * addr = (struct sockaddr_in *)rdma_get_peer_addr(cm_id_);
    if (addr) {
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &addr->sin_addr, ip, sizeof(ip));
        endpoint_ = std::string(ip) + ":" + std::to_string(ntohs(addr->sin_port));
    }

    // Setup QP
    if (!setup_qp(config)) {
        return false;
    }

    // Accept connection
    struct rdma_conn_param conn_param = {};
    conn_param.initiator_depth = 16;
    conn_param.responder_resources = 16;
    conn_param.rnr_retry_count = config.rnr_retry;

    if (rdma_accept(cm_id_, &conn_param) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to accept connection: %s\n", strerror(errno));
        return false;
    }

    connected_ = true;

    // Set min_rnr_timer to minimum value (1 = 0.01ms) to avoid 655ms RNR NAK delays.
    // Default min_rnr_timer (0 = 655.36ms) causes ~2 second spikes when sender
    // posts send before receiver posts recv (RNR condition).
    {
        struct ibv_qp_attr attr = {};
        attr.min_rnr_timer = 1;  // 0.01ms (value 1) instead of 655.36ms (value 0)
        if (ibv_modify_qp(qp_, &attr, IBV_QP_MIN_RNR_TIMER) != 0) {
            GGML_LOG_ERROR("[rdma_connection] Failed to set min_rnr_timer: %s\n", strerror(errno));
            // Non-fatal: continue with default timer
        } else {
            RDMA_LOG_DBG("[rdma_connection] Set min_rnr_timer=1 (0.01ms)\n");
        }
    }

    RDMA_LOG_DBG("[rdma_connection] Accepted connection from %s\n", endpoint_.c_str());
    return true;
}

void rdma_connection::disconnect() {
    if (recv_mr_) {
        ibv_dereg_mr(recv_mr_);
        recv_mr_ = nullptr;
    }

    if (send_mr_) {
        ibv_dereg_mr(send_mr_);
        send_mr_ = nullptr;
    }

    if (connected_ && cm_id_) {
        rdma_disconnect(cm_id_);
        connected_ = false;
    }

    if (qp_) {
        ibv_destroy_qp(qp_);
        qp_ = nullptr;
    }

    if (cq_) {
        ibv_destroy_cq(cq_);
        cq_ = nullptr;
    }

    if (comp_channel_) {
        ibv_destroy_comp_channel(comp_channel_);
        comp_channel_ = nullptr;
    }

    if (pd_) {
        ibv_dealloc_pd(pd_);
        pd_ = nullptr;
    }

    if (cm_id_) {
        rdma_destroy_id(cm_id_);
        cm_id_ = nullptr;
    }

    if (event_channel_) {
        rdma_destroy_event_channel(event_channel_);
        event_channel_ = nullptr;
    }

    RDMA_LOG_DBG("[rdma_connection] Disconnected from %s\n", endpoint_.c_str());
}

bool rdma_connection::setup_qp(const rdma_config & config) {
    // Allocate protection domain
    pd_ = ibv_alloc_pd(cm_id_->verbs);
    if (!pd_) {
        GGML_LOG_ERROR("[rdma_connection] Failed to allocate PD: %s\n", strerror(errno));
        return false;
    }

    // Create completion queue (no completion channel - using busy polling)
    cq_ = ibv_create_cq(cm_id_->verbs, config.cq_size, nullptr, nullptr, 0);
    if (!cq_) {
        GGML_LOG_ERROR("[rdma_connection] Failed to create CQ: %s\n", strerror(errno));
        return false;
    }

    // Create QP
    struct ibv_qp_init_attr qp_init_attr = {};
    qp_init_attr.send_cq = cq_;
    qp_init_attr.recv_cq = cq_;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = config.max_send_wr;
    qp_init_attr.cap.max_recv_wr = config.max_recv_wr;
    qp_init_attr.cap.max_send_sge = config.max_send_sge;
    qp_init_attr.cap.max_recv_sge = config.max_recv_sge;
    qp_init_attr.cap.max_inline_data = config.max_inline;

    if (rdma_create_qp(cm_id_, pd_, &qp_init_attr) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to create QP: %s\n", strerror(errno));
        return false;
    }
    qp_ = cm_id_->qp;

    // Setup receive buffer for control messages
    recv_buffer_.resize(config.recv_buf_size);
    recv_mr_ = ibv_reg_mr(pd_, recv_buffer_.data(), recv_buffer_.size(),
                          IBV_ACCESS_LOCAL_WRITE);
    if (!recv_mr_) {
        GGML_LOG_ERROR("[rdma_connection] Failed to register receive buffer: %s\n", strerror(errno));
        return false;
    }

    // Setup send buffer for control messages (when no external MR provided)
    send_buffer_.resize(config.recv_buf_size);
    send_mr_ = ibv_reg_mr(pd_, send_buffer_.data(), send_buffer_.size(),
                          IBV_ACCESS_LOCAL_WRITE);
    if (!send_mr_) {
        GGML_LOG_ERROR("[rdma_connection] Failed to register send buffer: %s\n", strerror(errno));
        return false;
    }

    RDMA_LOG_DBG("[rdma_connection] QP setup complete\n");
    return true;
}

bool rdma_connection::connect_qp() {
    struct rdma_conn_param conn_param = {};
    conn_param.initiator_depth = 16;
    conn_param.responder_resources = 16;
    conn_param.retry_count = config_.retry_count;
    conn_param.rnr_retry_count = config_.rnr_retry;

    if (rdma_connect(cm_id_, &conn_param) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to connect: %s\n", strerror(errno));
        return false;
    }

    // Wait for connection established event
    struct rdma_cm_event * event = nullptr;
    if (rdma_get_cm_event(event_channel_, &event) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to get CM event: %s\n", strerror(errno));
        return false;
    }

    if (event->event != RDMA_CM_EVENT_ESTABLISHED) {
        GGML_LOG_ERROR("[rdma_connection] Connection failed, event: %d\n", event->event);
        rdma_ack_cm_event(event);
        return false;
    }
    rdma_ack_cm_event(event);

    // Set min_rnr_timer to minimum value on client side too
    {
        struct ibv_qp_attr attr = {};
        attr.min_rnr_timer = 1;  // 0.01ms
        if (ibv_modify_qp(qp_, &attr, IBV_QP_MIN_RNR_TIMER) != 0) {
            GGML_LOG_ERROR("[rdma_connection] Failed to set min_rnr_timer: %s\n", strerror(errno));
        } else {
            RDMA_LOG_DBG("[rdma_connection] Set min_rnr_timer=1 (0.01ms)\n");
        }
    }

    return true;
}

struct ibv_mr * rdma_connection::register_memory(void * addr, size_t size, int access_flags) {
    if (!pd_) {
        GGML_LOG_ERROR("[rdma_connection] No protection domain\n");
        return nullptr;
    }

    struct ibv_mr * mr = ibv_reg_mr(pd_, addr, size, access_flags);
    if (!mr) {
        GGML_LOG_ERROR("[rdma_connection] Failed to register memory: %s\n", strerror(errno));
        return nullptr;
    }

    RDMA_LOG_DBG("[rdma_connection] Registered memory: addr=%p, size=%zu, lkey=0x%x, rkey=0x%x\n",
                 addr, size, mr->lkey, mr->rkey);
    return mr;
}

void rdma_connection::deregister_memory(struct ibv_mr * mr) {
    if (mr) {
        ibv_dereg_mr(mr);
    }
}

bool rdma_connection::send(const void * data, size_t size, struct ibv_mr * mr) {
    std::lock_guard<std::mutex> lock(send_mutex_);

    if (!connected_ || !qp_) {
        return false;
    }

    const uint8_t * src = static_cast<const uint8_t *>(data);
    size_t remaining = size;
    const size_t chunk_size = send_buffer_.size(); // Use internal buffer size as max chunk

    static constexpr size_t RDMA_SIGNAL_INTERVAL = 64;
    size_t unsignaled_count = 0;

    while (remaining > 0) {
        size_t send_size = remaining;

        struct ibv_sge sge = {};
        struct ibv_send_wr wr = {};
        struct ibv_send_wr * bad_wr = nullptr;

        if (mr) {
            sge.addr = (uint64_t)src;
            sge.length = send_size;
            sge.lkey = mr->lkey;
            wr.sg_list = &sge;
            wr.num_sge = 1;
        } else if (send_size <= config_.max_inline) {
            // Use inline data
            sge.addr = (uint64_t)src;
            sge.length = send_size;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.send_flags |= IBV_SEND_INLINE;
        } else if (chunk_size > 0 && send_mr_) {
            // Use internal send buffer, chunk if needed
            if (send_size > chunk_size) {
                send_size = chunk_size;
            }
            std::memcpy(send_buffer_.data(), src, send_size);
            sge.addr = (uint64_t)send_buffer_.data();
            sge.length = send_size;
            sge.lkey = send_mr_->lkey;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            if (send_size < remaining) {
                RDMA_LOG_DBG("[rdma_connection] Chunked send: %zu/%zu bytes\n", send_size, remaining);
            }
        } else {
            GGML_LOG_ERROR("[rdma_connection] Message too large for inline send without MR\n");
            GGML_LOG_ERROR("[rdma_connection]   size=%zu, max_inline=%u, send_buffer_.size()=%zu, send_mr_=%p\n",
                           send_size, config_.max_inline, send_buffer_.size(), (void*)send_mr_);
            return false;
        }

        wr.opcode = IBV_WR_SEND;
        bool is_last = (remaining == send_size);
        bool do_signal = RDMA_NO_SELECTIVE_SIGNAL || is_last || (++unsignaled_count >= RDMA_SIGNAL_INTERVAL);
        if (do_signal) {
            wr.send_flags |= IBV_SEND_SIGNALED;
        }

        if (ibv_post_send(qp_, &wr, &bad_wr) != 0) {
            GGML_LOG_ERROR("[rdma_connection] Failed to post send: %s\n", strerror(errno));
            return false;
        }

        if (do_signal) {
            if (!wait_for_completion(RDMA_TIMEOUT_MS)) {
                GGML_LOG_ERROR("[rdma_connection] Send completion timeout\n");
                return false;
            }
            unsignaled_count = 0;
        }

        stats_.bytes_sent += send_size;
        stats_.send_ops++;
        src += send_size;
        remaining -= send_size;
    }

    RDMA_LOG_DBG("[rdma_connection] Sent %zu bytes\n", size);
    return true;
}

bool rdma_connection::recv(void * data, size_t size, struct ibv_mr * mr, int timeout_ms) {
    if (!connected_ || !qp_) {
        return false;
    }

    // timeout_ms semantics:
    //   -1 (default): use RDMA_TIMEOUT_MS (30s)
    //    0: infinite wait (no timeout) — for server command loop idle wait
    //   >0: use specified value
    int effective_timeout;
    if (timeout_ms == 0) {
        effective_timeout = -1;  // negative = infinite in wait_for_completion
    } else if (timeout_ms > 0) {
        effective_timeout = timeout_ms;
    } else {
        effective_timeout = RDMA_TIMEOUT_MS;  // default (30s)
    }

    uint8_t * dst = static_cast<uint8_t *>(data);
    size_t remaining = size;
    const size_t chunk_size = recv_buffer_.size(); // Use internal buffer size as max chunk

    while (remaining > 0) {
        size_t recv_size = remaining;
        void * recv_addr = dst;
        struct ibv_mr * recv_mr_use = mr;

        if (mr) {
            // External MR: receive directly (no chunking needed, caller manages buffer)
            recv_size = remaining;
        } else if (chunk_size > 0 && recv_mr_) {
            // Use internal receive buffer, chunk if needed
            if (recv_size > chunk_size) {
                recv_size = chunk_size;
            }
            recv_addr = recv_buffer_.data();
            recv_mr_use = recv_mr_;
        } else {
            GGML_LOG_ERROR("[rdma_connection] No MR provided and message too large (%zu bytes) for internal buffer (%zu bytes)\n",
                           recv_size, recv_buffer_.size());
            return false;
        }

        // Post receive
        if (!post_recv(recv_addr, recv_size, recv_mr_use, 0)) {
            return false;
        }

        // Wait for completion
        if (!wait_for_completion(effective_timeout)) {
            GGML_LOG_ERROR("[rdma_connection] Receive completion timeout\n");
            return false;
        }

        // Copy data from internal buffer if needed
        if (!mr && recv_addr != dst) {
            std::memcpy(dst, recv_buffer_.data(), recv_size);
        }

        stats_.bytes_received += recv_size;
        stats_.recv_ops++;
        dst += recv_size;
        remaining -= recv_size;
    }

    RDMA_LOG_DBG("[rdma_connection] Received %zu bytes\n", size);
    return true;
}

bool rdma_connection::post_recv(void * data, size_t size, struct ibv_mr * mr, uint64_t wr_id) {
    if (!connected_ || !qp_) {
        return false;
    }

    struct ibv_sge sge = {};
    sge.addr = (uint64_t)data;
    sge.length = size;
    sge.lkey = mr->lkey;

    struct ibv_recv_wr wr = {};
    struct ibv_recv_wr * bad_wr = nullptr;
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_recv(qp_, &wr, &bad_wr) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to post receive: %s\n", strerror(errno));
        return false;
    }

    return true;
}

bool rdma_connection::rdma_write(const void * local_data, size_t size, struct ibv_mr * local_mr,
                                  const remote_memory_info & remote, bool signaled) {
    std::lock_guard<std::mutex> lock(send_mutex_);

    if (!connected_ || !qp_) {
        GGML_LOG_ERROR("[rdma_connection] rdma_write: not connected\n");
        return false;
    }

    RDMA_LOG_DBG("[rdma_connection] rdma_write: local=%p, size=%zu, lkey=0x%x, remote_addr=0x%lx, rkey=0x%x\n",
                 local_data, size, local_mr->lkey, (unsigned long)remote.addr, remote.rkey);

    struct ibv_sge sge = {};
    sge.addr = (uint64_t)local_data;
    sge.length = size;
    sge.lkey = local_mr->lkey;

    struct ibv_send_wr wr = {};
    struct ibv_send_wr * bad_wr = nullptr;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = remote.addr;
    wr.wr.rdma.rkey = remote.rkey;
    if (signaled) {
        wr.send_flags = IBV_SEND_SIGNALED;
    }

    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    if (ret != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to post RDMA write: %s (ret=%d)\n", strerror(errno), ret);
        return false;
    }

    if (signaled) {
        if (!wait_for_completion(RDMA_TIMEOUT_MS)) {
            GGML_LOG_ERROR("[rdma_connection] RDMA write completion timeout: size=%zu, remote_addr=0x%lx, rkey=0x%x\n",
                           size, (unsigned long)remote.addr, remote.rkey);
            return false;
        }
    }

    stats_.bytes_sent += size;
    stats_.rdma_writes++;
    RDMA_LOG_DBG("[rdma_connection] RDMA write done: %zu bytes to remote 0x%lx\n", size, (unsigned long)remote.addr);
    return true;
}

bool rdma_connection::rdma_read(void * local_data, size_t size, struct ibv_mr * local_mr,
                                 const remote_memory_info & remote, bool signaled) {
    std::lock_guard<std::mutex> lock(send_mutex_);

    if (!connected_ || !qp_) {
        return false;
    }

    struct ibv_sge sge = {};
    sge.addr = (uint64_t)local_data;
    sge.length = size;
    sge.lkey = local_mr->lkey;

    struct ibv_send_wr wr = {};
    struct ibv_send_wr * bad_wr = nullptr;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = remote.addr;
    wr.wr.rdma.rkey = remote.rkey;
    if (signaled) {
        wr.send_flags = IBV_SEND_SIGNALED;
    }

    if (ibv_post_send(qp_, &wr, &bad_wr) != 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to post RDMA read: %s\n", strerror(errno));
        return false;
    }

    if (signaled) {
        if (!wait_for_completion(RDMA_TIMEOUT_MS)) {
            GGML_LOG_ERROR("[rdma_connection] RDMA read completion timeout\n");
            return false;
        }
    }

    stats_.bytes_received += size;
    stats_.rdma_reads++;
    RDMA_LOG_DBG("[rdma_connection] RDMA read: %zu bytes from remote 0x%lx\n", size, remote.addr);
    return true;
}

int rdma_connection::poll_cq(int max_entries) {
    std::lock_guard<std::mutex> lock(poll_mutex_);

    if (!cq_) {
        return -1;
    }

    std::vector<struct ibv_wc> wc(max_entries);
    int n = ibv_poll_cq(cq_, max_entries, wc.data());

    if (n < 0) {
        GGML_LOG_ERROR("[rdma_connection] Failed to poll CQ: %s\n", strerror(errno));
        return -1;
    }

    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            GGML_LOG_ERROR("[rdma_connection] Work completion error: %s (opcode=%d)\n",
                          ibv_wc_status_str(wc[i].status), wc[i].opcode);
        }
    }

    return n;
}

bool rdma_connection::wait_for_completion(int timeout_ms) {
    if (!cq_) {
        return false;
    }

    struct ibv_wc wc = {};
    auto start = std::chrono::steady_clock::now();
    int poll_count = 0;
    int64_t last_progress_ms = 0;

    while (true) {
        int n = ibv_poll_cq(cq_, 1, &wc);
        if (n > 0) {
            if (wc.status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("[rdma_connection] Work completion error: status=%s, opcode=%d, vendor_err=0x%x\n",
                               ibv_wc_status_str(wc.status), wc.opcode, wc.vendor_err);
                return false;
            }
            RDMA_LOG_DBG("[rdma_connection] CQ completion: opcode=%d, bytes=%u\n", wc.opcode, wc.byte_len);
            return true;
        }
        if (n < 0) {
            GGML_LOG_ERROR("[rdma_connection] Failed to poll CQ: %s\n", strerror(errno));
            return false;
        }
        (void)poll_count;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        // For infinite-wait operations (e.g., server waiting for next client command),
        // sleep after initial busy-poll period to avoid 100% CPU usage.
        // Regular RDMA ops (timeout >= 0) always busy-poll for lowest latency.
        if (timeout_ms < 0 && elapsed > 1000) {
            usleep(1000);  // 1ms sleep — acceptable latency for idle command wait
        }

        // Progress log every 10 seconds for long waits (skip for infinite-wait idle loops)
        if (timeout_ms >= 0 && elapsed > 10000 && elapsed - last_progress_ms >= 10000) {
            GGML_LOG_WARN("[rdma_connection] Still waiting for completion: %lld ms elapsed (timeout=%d ms)\n",
                          (long long)elapsed, timeout_ms);
            last_progress_ms = elapsed;
        }

        if (timeout_ms >= 0 && elapsed > timeout_ms) {
            GGML_LOG_ERROR("[rdma_connection] Completion timeout after %d ms\n", timeout_ms);
            return false;
        }
    }
}

remote_memory_info rdma_connection::get_local_mr_info(struct ibv_mr * mr) const {
    remote_memory_info info = {};
    if (mr) {
        info.addr = (uint64_t)mr->addr;
        info.rkey = mr->rkey;
        info.size = mr->length;
    }
    return info;
}

void rdma_connection::reset_stats() {
    stats_.bytes_sent = 0;
    stats_.bytes_received = 0;
    stats_.rdma_writes = 0;
    stats_.rdma_reads = 0;
    stats_.send_ops = 0;
    stats_.recv_ops = 0;
}

// rdma_connection_manager implementation

rdma_connection_manager::rdma_connection_manager() {
}

rdma_connection_manager::~rdma_connection_manager() {
    stop_server();
}

std::shared_ptr<rdma_connection> rdma_connection_manager::get_connection(const std::string & key) {
    std::lock_guard<std::mutex> lock(connections_mutex_);

    // Extract endpoint from key (strip "#device_idx" suffix if present)
    std::string endpoint = key;
    auto hash_pos = key.find('#');
    if (hash_pos != std::string::npos) {
        endpoint = key.substr(0, hash_pos);
    }

    // Determine cache key: shared mode (default) uses endpoint only, per-device uses full key.
    // Per-device connections create separate QPs per device, which can cause RNIC MTT cache
    // overflow on ConnectX-4 with large models (>40GB total MR). Use GGML_RDMA_PER_DEVICE_CONN=1
    // to enable per-device connections on newer NICs (ConnectX-6+) with larger MTT caches.
    static bool per_device_conn = (std::getenv("GGML_RDMA_PER_DEVICE_CONN") != nullptr);
    const std::string & cache_key = per_device_conn ? key : endpoint;

    // Check if connection already exists
    auto it = connections_.find(cache_key);
    if (it != connections_.end()) {
        if (it->second && it->second->is_connected()) {
            return it->second;
        }
        connections_.erase(it);
    }

    // Parse endpoint
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("[rdma_connection_manager] Invalid endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    // Create new connection
    auto conn = std::make_shared<rdma_connection>();
    if (!conn->connect(host.c_str(), port)) {
        return nullptr;
    }

    connections_[cache_key] = conn;
    return conn;
}

bool rdma_connection_manager::start_server(const char * host, int port, const rdma_config & config) {
    if (server_running_) {
        return false;
    }

    server_config_ = config;

    // Create event channel
    server_channel_ = rdma_create_event_channel();
    if (!server_channel_) {
        GGML_LOG_ERROR("[rdma_connection_manager] Failed to create event channel: %s\n", strerror(errno));
        return false;
    }

    // Create CM ID for server
    if (rdma_create_id(server_channel_, &server_id_, nullptr, RDMA_PS_TCP) != 0) {
        GGML_LOG_ERROR("[rdma_connection_manager] Failed to create server CM ID: %s\n", strerror(errno));
        rdma_destroy_event_channel(server_channel_);
        server_channel_ = nullptr;
        return false;
    }

    // Bind to address
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        GGML_LOG_ERROR("[rdma_connection_manager] Invalid host address: %s\n", host);
        rdma_destroy_id(server_id_);
        rdma_destroy_event_channel(server_channel_);
        server_id_ = nullptr;
        server_channel_ = nullptr;
        return false;
    }

    if (rdma_bind_addr(server_id_, (struct sockaddr *)&addr) != 0) {
        GGML_LOG_ERROR("[rdma_connection_manager] Failed to bind: %s\n", strerror(errno));
        rdma_destroy_id(server_id_);
        rdma_destroy_event_channel(server_channel_);
        server_id_ = nullptr;
        server_channel_ = nullptr;
        return false;
    }

    // Listen
    if (rdma_listen(server_id_, 10) != 0) {
        GGML_LOG_ERROR("[rdma_connection_manager] Failed to listen: %s\n", strerror(errno));
        rdma_destroy_id(server_id_);
        rdma_destroy_event_channel(server_channel_);
        server_id_ = nullptr;
        server_channel_ = nullptr;
        return false;
    }

    server_running_ = true;
    GGML_LOG_INFO("[rdma_connection_manager] Server listening on %s:%d\n", host, port);
    return true;
}

void rdma_connection_manager::stop_server() {
    if (!server_running_) {
        return;
    }

    server_running_ = false;

    if (server_id_) {
        rdma_destroy_id(server_id_);
        server_id_ = nullptr;
    }

    if (server_channel_) {
        rdma_destroy_event_channel(server_channel_);
        server_channel_ = nullptr;
    }
}

std::shared_ptr<rdma_connection> rdma_connection_manager::accept_connection() {
    if (!server_running_ || !server_channel_) {
        return nullptr;
    }

    while (true) {
        // Wait for connection request
        struct rdma_cm_event * event = nullptr;
        if (rdma_get_cm_event(server_channel_, &event) != 0) {
            if (!server_running_) {
                // Server was stopped (e.g. signal handler)
                return nullptr;
            }
            GGML_LOG_ERROR("[rdma_connection_manager] Failed to get CM event: %s\n", strerror(errno));
            return nullptr;
        }

        // Handle DISCONNECT from previous client: call rdma_disconnect() to transition
        // QP to ERROR state, which flushes pending recv WRs and unblocks the command
        // loop thread (prevents thread leak).
        if (event->event == RDMA_CM_EVENT_DISCONNECTED) {
            struct rdma_cm_id * disconnected_id = event->id;  // save before ack frees event
            GGML_LOG_INFO("[rdma_connection_manager] Client disconnected (CM event), triggering QP error transition\n");
            rdma_ack_cm_event(event);
            rdma_disconnect(disconnected_id);
            continue;
        }

        // Skip other non-connect events
        if (event->event != RDMA_CM_EVENT_CONNECT_REQUEST) {
            RDMA_LOG_DBG("[rdma_connection_manager] Skipping CM event: %s (%d)\n",
                         rdma_event_str(event->event), event->event);
            rdma_ack_cm_event(event);
            continue;
        }

        struct rdma_cm_id * client_id = event->id;
        rdma_ack_cm_event(event);

        // Create connection and accept
        auto conn = std::make_shared<rdma_connection>();
        if (!conn->accept(client_id, server_config_)) {
            rdma_destroy_id(client_id);
            return nullptr;
        }

        // Wait for ESTABLISHED event for this specific connection.
        // With multi-connection setups, we may receive events for other connections
        // (ESTABLISHED, DISCONNECTED, etc.) - skip those and continue waiting.
        bool established = false;
        while (true) {
            if (rdma_get_cm_event(server_channel_, &event) != 0) {
                if (!server_running_) {
                    return nullptr;
                }
                GGML_LOG_ERROR("[rdma_connection_manager] Failed to get established event: %s\n", strerror(errno));
                return nullptr;
            }

            if (event->event == RDMA_CM_EVENT_ESTABLISHED && event->id == client_id) {
                rdma_ack_cm_event(event);
                established = true;
                break;
            }

            // Check if this connection was rejected before acking
            bool is_failure = (event->id == client_id &&
                (event->event == RDMA_CM_EVENT_REJECTED ||
                 event->event == RDMA_CM_EVENT_CONNECT_ERROR));

            // Handle DISCONNECT for other connections (trigger QP error transition)
            if (event->event == RDMA_CM_EVENT_DISCONNECTED && event->id != client_id) {
                struct rdma_cm_id * disconnected_id = event->id;
                GGML_LOG_INFO("[rdma_connection_manager] Other client disconnected (CM event) during ESTABLISHED wait\n");
                rdma_ack_cm_event(event);
                rdma_disconnect(disconnected_id);
                continue;
            }

            // Skip other events for other connections
            RDMA_LOG_DBG("[rdma_connection_manager] Skipping CM event while waiting for ESTABLISHED: %s (%d) for id %p (want %p)\n",
                         rdma_event_str(event->event), event->event, (void*)event->id, (void*)client_id);
            rdma_ack_cm_event(event);

            // If this connection was rejected, give up on it
            if (is_failure) {
                break;
            }
        }

        if (established) {
            return conn;
        }
        // Connection failed, try accepting again
    }
}

} // namespace ggml_rdma
