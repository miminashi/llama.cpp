// Simple RDMA Client Test - minimal test for debugging
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rdma.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("Usage: %s <endpoint>\n", argv[0]);
        return 1;
    }
    const char * endpoint = argv[1];

    printf("[1] Testing GPUDirect availability...\n");
    bool gdr = ggml_backend_rdma_gdr_available();
    printf("    GPUDirect RDMA available: %s\n\n", gdr ? "yes" : "no");

    printf("[2] Connecting to %s...\n", endpoint);
    ggml_backend_t backend = ggml_backend_rdma_init(endpoint, 0);
    if (!backend) {
        fprintf(stderr, "ERROR: Failed to connect\n");
        return 1;
    }
    printf("    Connected! Backend: %s\n\n", ggml_backend_name(backend));

    printf("[3] Getting device memory...\n");
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_rdma_get_device_memory(endpoint, 0, &free_mem, &total_mem);
    printf("    Free: %.2f GB, Total: %.2f GB\n\n",
           (double)free_mem / (1024.0*1024.0*1024.0),
           (double)total_mem / (1024.0*1024.0*1024.0));

    printf("[4] Getting buffer type...\n");
    ggml_backend_buffer_type_t buft = ggml_backend_rdma_buffer_type(endpoint, 0);
    if (!buft) {
        fprintf(stderr, "ERROR: Failed to get buffer type\n");
        ggml_backend_free(backend);
        return 1;
    }
    printf("    Buffer type: %s\n\n", ggml_backend_buft_name(buft));

    printf("[5] Creating simple tensor...\n");
    struct ggml_init_params params = {
        ggml_tensor_overhead() * 2,
        nullptr,
        true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_set_name(tensor, "test");
    printf("    Created tensor: %s\n\n", tensor->name);

    printf("[6] Allocating buffer on remote...\n");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "ERROR: Failed to allocate buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    printf("    Buffer allocated: %zu bytes\n\n", ggml_backend_buffer_get_size(buffer));

    printf("[7] Setting tensor data...\n");
    float data[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    ggml_backend_tensor_set(tensor, data, 0, sizeof(data));
    printf("    Set tensor data OK\n\n");

    printf("[8] Getting tensor data...\n");
    float result[16] = {0};
    ggml_backend_tensor_get(tensor, result, 0, sizeof(result));
    printf("    Got tensor data: [%.1f, %.1f, %.1f, ...]\n\n", result[0], result[1], result[2]);

    printf("[9] Cleanup...\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    printf("    Done!\n");

    return 0;
}
