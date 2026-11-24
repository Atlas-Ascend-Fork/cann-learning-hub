#pragma once

#include <acl/acl.h>
#include <cann_ops_sparse.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace acl_demo {
inline void check_acl(aclError code, const char* expression) {
    if (code != ACL_SUCCESS) throw std::runtime_error(std::string(expression) + " failed, aclError=" + std::to_string(code));
}
inline void check_sparse(aclsparseStatus_t code, const char* expression) {
    if (code != ACL_SPARSE_STATUS_SUCCESS) throw std::runtime_error(std::string(expression) + " failed, aclsparseStatus=" + std::to_string(code));
}
#define ACL_CHECK(expr) ::acl_demo::check_acl((expr), #expr)
#define SPARSE_CHECK(expr) ::acl_demo::check_sparse((expr), #expr)
inline std::size_t numel(const std::vector<int64_t>& shape) { std::size_t n = 1; for (auto d : shape) n *= static_cast<std::size_t>(d); return n; }
class DeviceBuffer {
public:
    DeviceBuffer() = default; explicit DeviceBuffer(std::size_t n) { allocate(n); } ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete; DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(std::size_t n) { reset(); bytes_ = n; if (n) ACL_CHECK(aclrtMalloc(&data_, n, ACL_MEM_MALLOC_HUGE_FIRST)); }
    void reset() noexcept { if (data_) aclrtFree(data_); data_ = nullptr; bytes_ = 0; }
    void* data() const { return data_; } std::size_t bytes() const { return bytes_; }
private: void* data_ = nullptr; std::size_t bytes_ = 0;
};
class Runtime {
public:
    explicit Runtime(int device) : device_(device) {
        try {
            ACL_CHECK(aclInit(nullptr));
            initialized_ = true;
            ACL_CHECK(aclrtSetDevice(device_));
            device_set_ = true;
            ACL_CHECK(aclrtCreateContext(&context_, device_));
            ACL_CHECK(aclrtSetCurrentContext(context_));
            ACL_CHECK(aclrtCreateStream(&stream_));
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~Runtime() { cleanup(); }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    aclrtStream stream() const { return stream_; }
private:
    void cleanup() noexcept {
        if (stream_) aclrtDestroyStream(stream_);
        if (context_) aclrtDestroyContext(context_);
        if (device_set_) aclrtResetDevice(device_);
        if (initialized_) aclFinalize();
    }
    int device_;
    bool initialized_ = false;
    bool device_set_ = false;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
};
inline void copy_to_device(const DeviceBuffer& d,const void* h,std::size_t n){ ACL_CHECK(aclrtMemcpy(d.data(),d.bytes(),h,n,ACL_MEMCPY_HOST_TO_DEVICE)); }
inline void copy_to_host(void* h,std::size_t n,const DeviceBuffer& d){ ACL_CHECK(aclrtMemcpy(h,n,d.data(),d.bytes(),ACL_MEMCPY_DEVICE_TO_HOST)); }
inline void* allocate_workspace(std::size_t n){ void* p=nullptr; if(n) ACL_CHECK(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST)); return p; }
inline void free_workspace(void* p){ if(p) aclrtFree(p); }
} // namespace acl_demo
