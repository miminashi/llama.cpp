message(STATUS "Using RDMA backend")

# Find required RDMA libraries
find_library(IBVERBS_LIBRARY NAMES ibverbs)
find_library(RDMACM_LIBRARY NAMES rdmacm)
find_path(IBVERBS_INCLUDE_DIR NAMES infiniband/verbs.h)
find_path(RDMACM_INCLUDE_DIR NAMES rdma/rdma_cma.h)

if(NOT IBVERBS_LIBRARY OR NOT RDMACM_LIBRARY)
    message(FATAL_ERROR "RDMA libraries (libibverbs, librdmacm) not found. "
                        "Please install rdma-core: apt install libibverbs-dev librdmacm-dev")
endif()

if(NOT IBVERBS_INCLUDE_DIR OR NOT RDMACM_INCLUDE_DIR)
    message(FATAL_ERROR "RDMA header files not found. "
                        "Please install rdma-core development files.")
endif()

message(STATUS "  ibverbs library: ${IBVERBS_LIBRARY}")
message(STATUS "  rdmacm library: ${RDMACM_LIBRARY}")

# Check for CUDA support (optional, for GPUDirect RDMA)
set(GGML_RDMA_CUDA OFF)
if(GGML_CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        set(GGML_RDMA_CUDA ON)
        message(STATUS "  CUDA support: enabled (GPUDirect RDMA available)")
    endif()
endif()

# Check for nvidia-peermem availability (runtime check will be done in code)
if(GGML_RDMA_CUDA)
    execute_process(
        COMMAND modinfo nvidia-peermem
        RESULT_VARIABLE PEERMEM_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )
    if(PEERMEM_RESULT EQUAL 0)
        message(STATUS "  nvidia-peermem: module available")
    else()
        message(STATUS "  nvidia-peermem: module not found (GPUDirect RDMA may not work)")
    endif()
endif()

# Source files
set(GGML_RDMA_SOURCES
    ggml-rdma.cpp
    rdma-transport.cpp
    rdma-memory.cpp
    rdma-gdr.cpp
)

ggml_add_backend_library(ggml-rdma ${GGML_RDMA_SOURCES})

target_include_directories(ggml-rdma PRIVATE
    ${IBVERBS_INCLUDE_DIR}
    ${RDMACM_INCLUDE_DIR}
)

target_link_libraries(ggml-rdma PRIVATE
    ${IBVERBS_LIBRARY}
    ${RDMACM_LIBRARY}
)

if(GGML_RDMA_CUDA)
    target_compile_definitions(ggml-rdma PRIVATE GGML_RDMA_CUDA)
    target_link_libraries(ggml-rdma PRIVATE CUDA::cuda_driver CUDA::cudart)
    target_include_directories(ggml-rdma PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
