// RDMA Client Test Program
// Tests basic RDMA connectivity and tensor operations

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rdma.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static void print_usage(const char * prog) {
    printf("Usage: %s <endpoint> [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  -d <device>   Device index on remote server (default: 0)\n");
    printf("  -s <size>     Tensor size for test (default: 1024)\n");
    printf("  -h            Show this help\n");
    printf("\n");
    printf("Example: %s 192.168.100.2:50051 -d 0 -s 4096\n", prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char * endpoint = argv[1];
    uint32_t device = 0;
    int tensor_size = 1024;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            tensor_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("=== RDMA Client Test ===\n");
    printf("Endpoint: %s\n", endpoint);
    printf("Device: %u\n", device);
    printf("Tensor size: %d\n\n", tensor_size);

    // Check GPUDirect availability
    bool gdr = ggml_backend_rdma_gdr_available();
    printf("GPUDirect RDMA available: %s\n\n", gdr ? "yes" : "no");

    // Step 1: Connect to remote server
    printf("[1] Connecting to remote server...\n");
    ggml_backend_t backend = ggml_backend_rdma_init(endpoint, device);
    if (!backend) {
        fprintf(stderr, "ERROR: Failed to connect to %s\n", endpoint);
        return 1;
    }
    printf("    Connected successfully!\n");
    printf("    Backend name: %s\n\n", ggml_backend_name(backend));

    // Step 2: Get device memory info
    printf("[2] Querying device memory...\n");
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_rdma_get_device_memory(endpoint, device, &free_mem, &total_mem);
    printf("    Free: %.2f GB\n", (double)free_mem / (1024.0 * 1024.0 * 1024.0));
    printf("    Total: %.2f GB\n\n", (double)total_mem / (1024.0 * 1024.0 * 1024.0));

    // Step 3: Get buffer type
    printf("[3] Getting buffer type...\n");
    ggml_backend_buffer_type_t buft = ggml_backend_rdma_buffer_type(endpoint, device);
    if (!buft) {
        fprintf(stderr, "ERROR: Failed to get buffer type\n");
        ggml_backend_free(backend);
        return 1;
    }
    printf("    Buffer type: %s\n\n", ggml_backend_buft_name(buft));

    // Step 4: Create and allocate tensors
    printf("[4] Creating tensors...\n");
    struct ggml_init_params params = {
        .mem_size   = ggml_tensor_overhead() * 4,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");

    // Allocate buffer on remote server
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "ERROR: Failed to allocate buffer on remote server\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    printf("    Buffer allocated: %zu bytes\n\n", ggml_backend_buffer_get_size(buffer));

    // Step 5: Set tensor data
    printf("[5] Setting tensor data via RDMA...\n");
    std::vector<float> data_a(tensor_size);
    std::vector<float> data_b(tensor_size);
    for (int i = 0; i < tensor_size; i++) {
        data_a[i] = (float)i * 0.001f;
        data_b[i] = (float)(tensor_size - i) * 0.001f;
    }

    ggml_backend_tensor_set(a, data_a.data(), 0, tensor_size * sizeof(float));
    ggml_backend_tensor_set(b, data_b.data(), 0, tensor_size * sizeof(float));
    printf("    Set tensor 'a' and 'b' data\n\n");

    // Step 6: Get tensor data back
    printf("[6] Getting tensor data via RDMA...\n");
    std::vector<float> data_a_check(tensor_size);
    std::vector<float> data_b_check(tensor_size);

    ggml_backend_tensor_get(a, data_a_check.data(), 0, tensor_size * sizeof(float));
    ggml_backend_tensor_get(b, data_b_check.data(), 0, tensor_size * sizeof(float));

    // Verify data
    bool match_a = true, match_b = true;
    for (int i = 0; i < tensor_size && (match_a || match_b); i++) {
        if (match_a && fabsf(data_a[i] - data_a_check[i]) > 1e-6f) {
            match_a = false;
            printf("    ERROR: Tensor 'a' mismatch at index %d: expected %f, got %f\n",
                   i, data_a[i], data_a_check[i]);
        }
        if (match_b && fabsf(data_b[i] - data_b_check[i]) > 1e-6f) {
            match_b = false;
            printf("    ERROR: Tensor 'b' mismatch at index %d: expected %f, got %f\n",
                   i, data_b[i], data_b_check[i]);
        }
    }

    if (match_a && match_b) {
        printf("    Data verification: PASSED\n\n");
    } else {
        printf("    Data verification: FAILED\n\n");
    }

    // Step 7: Get RDMA statistics
    printf("[7] RDMA Statistics:\n");
    ggml_rdma_stats_t stats;
    ggml_backend_rdma_get_stats(backend, &stats);
    printf("    Bytes sent:     %lu\n", stats.bytes_sent);
    printf("    Bytes received: %lu\n", stats.bytes_received);
    printf("    RDMA writes:    %lu\n", stats.rdma_writes);
    printf("    RDMA reads:     %lu\n", stats.rdma_reads);
    printf("    Send ops:       %lu\n", stats.send_ops);
    printf("    Recv ops:       %lu\n\n", stats.recv_ops);

    // Cleanup
    printf("[8] Cleaning up...\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    printf("    Done!\n\n");

    printf("=== Test Complete ===\n");
    return (match_a && match_b) ? 0 : 1;
}
