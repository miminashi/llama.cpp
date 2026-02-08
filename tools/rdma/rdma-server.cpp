#include "ggml-rdma.h"

#ifdef _WIN32
#  define NOMINMAX
#  define DIRECTORY_SEPARATOR '\\'
#  include <windows.h>
#else
#  define DIRECTORY_SEPARATOR '/'
#  include <unistd.h>
#  include <sys/stat.h>
#  include <csignal>
#endif

#include <string>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <thread>
#include <regex>

static volatile sig_atomic_t g_server_shutdown = 0;
static decltype(ggml_backend_rdma_stop_server) * g_stop_server_fn = nullptr;

static void signal_handler(int signum) {
    g_server_shutdown = 1;
    // Stop the server to unblock accept_connection()
    if (g_stop_server_fn) {
        g_stop_server_fn();
    }
    (void)signum;
}

static bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    // Windows implementation would go here
    return false;
#else
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1;
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat sub_info;

        if (stat(subpath.c_str(), &sub_info) == 0) {
            if (!S_ISDIR(sub_info.st_mode)) {
                return false;
            }
        } else {
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }
        pos_slash += 1;
    }

    return true;
#endif
}

static std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    } else {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || defined(__OpenBSD__)
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        } else {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        }
#elif defined(__APPLE__)
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
        cache_directory = std::getenv("LOCALAPPDATA");
#else
#  error Unknown architecture
#endif
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

struct rdma_server_params {
    std::string              host        = "0.0.0.0";
    int                      port        = 50051;
    bool                     use_cache   = false;
    int                      n_threads   = std::max(1U, std::thread::hardware_concurrency()/2);
    std::vector<std::string> devices;
};

static void print_usage(int /*argc*/, char ** argv, rdma_server_params params) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "GPUDirect RDMA server for distributed inference\n\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                       show this help message and exit\n");
    fprintf(stderr, "  -t, --threads N                  number of threads for the CPU device (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -d, --device <dev1,dev2,...>     comma-separated list of devices\n");
    fprintf(stderr, "  -H, --host HOST                  host to bind to (default: %s)\n", params.host.c_str());
    fprintf(stderr, "  -p, --port PORT                  port to bind to (default: %d)\n", params.port);
    fprintf(stderr, "  -c, --cache                      enable local file cache\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Environment variables:\n");
    fprintf(stderr, "  GGML_RDMA_DEBUG=1                enable debug output\n");
    fprintf(stderr, "\n");
}

static bool rdma_server_params_parse(int argc, char ** argv, rdma_server_params & params) {
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "-H" || arg == "--host") {
            if (++i >= argc) {
                return false;
            }
            params.host = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                return false;
            }
            params.n_threads = std::stoi(argv[i]);
            if (params.n_threads <= 0) {
                fprintf(stderr, "error: invalid number of threads: %d\n", params.n_threads);
                return false;
            }
        } else if (arg == "-d" || arg == "--device") {
            if (++i >= argc) {
                return false;
            }
            const std::regex regex{ R"([,/]+)" };
            std::string dev_str = argv[i];
            std::sregex_token_iterator iter(dev_str.begin(), dev_str.end(), regex, -1);
            std::sregex_token_iterator end;
            for ( ; iter != end; ++iter) {
                try {
                    params.devices.push_back(*iter);
                } catch (const std::exception & ) {
                    fprintf(stderr, "error: invalid device: %s\n", iter->str().c_str());
                    return false;
                }
            }
        } else if (arg == "-p" || arg == "--port") {
            if (++i >= argc) {
                return false;
            }
            params.port = std::stoi(argv[i]);
            if (params.port <= 0 || params.port > 65535) {
                return false;
            }
        } else if (arg == "-c" || arg == "--cache") {
            params.use_cache = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(1);
        }
    }
    return true;
}

static std::vector<ggml_backend_dev_t> get_devices(const rdma_server_params & params) {
    std::vector<ggml_backend_dev_t> devices;
    if (!params.devices.empty()) {
        for (auto device : params.devices) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
            if (dev) {
                devices.push_back(dev);
            } else {
                fprintf(stderr, "error: unknown device: %s\n", device.c_str());
                fprintf(stderr, "available devices:\n");
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    auto * d = ggml_backend_dev_get(i);
                    size_t free, total;
                    ggml_backend_dev_memory(d, &free, &total);
                    fprintf(stderr, "  %s: %s (%zu MiB, %zu MiB free)\n",
                            ggml_backend_dev_name(d), ggml_backend_dev_description(d),
                            total / 1024 / 1024, free / 1024 / 1024);
                }
                return {};
            }
        }
    }

    // Try non-CPU devices first (prefer GPU)
    if (devices.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                devices.push_back(dev);
            }
        }
    }

    // If there are no accelerators, fallback to CPU device
    if (devices.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (dev) {
            devices.push_back(dev);
        }
    }

    return devices;
}

int main(int argc, char * argv[]) {
    // Setup signal handlers for graceful shutdown
#ifndef _WIN32
    struct sigaction sa = {};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
#endif

    ggml_backend_load_all();

    rdma_server_params params;
    if (!rdma_server_params_parse(argc, argv, params)) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }

    if (params.host != "127.0.0.1" && params.host != "localhost") {
        fprintf(stderr, "\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "WARNING: Host ('%s') is != '127.0.0.1'\n", params.host.c_str());
        fprintf(stderr, "         Never expose the RDMA server to an open network!\n");
        fprintf(stderr, "         This is an experimental feature and is not secure!\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
    }

    auto devices = get_devices(params);
    if (devices.empty()) {
        fprintf(stderr, "No devices found\n");
        return 1;
    }

    std::string endpoint = params.host + ":" + std::to_string(params.port);
    const char * cache_dir = nullptr;
    std::string cache_dir_str;
    if (params.use_cache) {
        cache_dir_str = fs_get_cache_directory() + "rdma/";
        if (!fs_create_directory_with_parents(cache_dir_str)) {
            fprintf(stderr, "Failed to create cache directory: %s\n", cache_dir_str.c_str());
            return 1;
        }
        cache_dir = cache_dir_str.c_str();
    }

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RDMA");
    if (!reg) {
        fprintf(stderr, "Failed to find RDMA backend\n");
        fprintf(stderr, "Make sure llama.cpp was compiled with -DGGML_RDMA=ON\n");
        return 1;
    }

    auto start_server_fn = (decltype(ggml_backend_rdma_start_server)*) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rdma_start_server");
    if (!start_server_fn) {
        fprintf(stderr, "Failed to obtain RDMA backend start server function\n");
        return 1;
    }

    // Get stop_server function for signal handler
    g_stop_server_fn = (decltype(ggml_backend_rdma_stop_server)*) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rdma_stop_server");

    start_server_fn(endpoint.c_str(), cache_dir, params.n_threads, devices.size(), devices.data());
    return 0;
}
