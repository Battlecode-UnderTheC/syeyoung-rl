// The C++ runtime's view of the compiled model: XLA programs exported from JAX
// (dragons_rl/export.py, a build step) run through a PJRT plugin (the CUDA
// plugin jaxlib ships). No Python at run time.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pjrt_c_api.h"

namespace rt {

enum class DType : uint8_t { F32 = 0, S32 = 1, U8 = 2, F16 = 3, PRED = 4 };
size_t dtype_size(DType t);

struct TensorSpec {
  std::string name;
  DType type;
  std::vector<int64_t> dims;
  size_t bytes() const;
};

// One loaded plugin and its client (all local devices).
class Runtime {
 public:
  explicit Runtime(const std::string& plugin_path);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  int num_devices() const { return static_cast<int>(devices_.size()); }
  PJRT_Device* device(int i) const { return devices_.at(i); }
  const PJRT_Api* api() const { return api_; }
  PJRT_Client* client() const { return client_; }
  void check(PJRT_Error* e, const char* what) const;   // throws with the plugin's message
  void await(PJRT_Event* e) const;                      // waits, destroys
  // pins host memory for direct transfers (false if the plugin cannot)
  bool dma_map(void* data, size_t size) const;
  void dma_unmap(void* data) const;

 private:
  void* handle_ = nullptr;
  const PJRT_Api* api_ = nullptr;
  PJRT_Client* client_ = nullptr;
  std::vector<PJRT_Device*> devices_;
};

// A device buffer (owned).
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Runtime* rt, PJRT_Buffer* b) : rt_(rt), b_(b) {}
  Buffer(Buffer&& o) noexcept : rt_(o.rt_), b_(o.b_) { o.b_ = nullptr; }
  Buffer& operator=(Buffer&& o) noexcept;
  Buffer(const Buffer&) = delete;
  ~Buffer();
  PJRT_Buffer* get() const { return b_; }
  explicit operator bool() const { return b_ != nullptr; }
  // copies the whole buffer into dst (dst_bytes must match)
  void download(void* dst, size_t dst_bytes) const;

 private:
  const Runtime* rt_ = nullptr;
  PJRT_Buffer* b_ = nullptr;
};

// host -> device (the host memory may be reused once this returns)
Buffer upload(const Runtime& rt, int device, const TensorSpec& spec, const void* data);

// Several transfers in flight at once: start them all, then wait for all.
class Transfers {
 public:
  explicit Transfers(const Runtime& rt) : rt_(rt) {}
  ~Transfers();
  // host -> device; the host memory must stay put until wait()
  Buffer upload(int device, const TensorSpec& spec, const void* data);
  // device -> host; dst must stay put until wait()
  void download(const Buffer& src, void* dst, size_t bytes);
  void wait();

 private:
  const Runtime& rt_;
  std::vector<PJRT_Event*> events_;
};

// One exported program, compiled for one device.
class Program {
 public:
  Program(const Runtime& rt, const std::string& dir, const std::string& name, int device);
  ~Program();
  const std::vector<TensorSpec>& inputs() const { return inputs_; }
  const std::vector<TensorSpec>& outputs() const { return outputs_; }
  int input(const std::string& name) const;           // index, throws if absent
  int output(const std::string& name) const;
  int device() const { return device_; }
  // runs on this program's device; args in input order (not donated)
  std::vector<Buffer> run(const std::vector<PJRT_Buffer*>& args) const;

 private:
  const Runtime& rt_;
  int device_;
  PJRT_LoadedExecutable* exe_ = nullptr;
  std::vector<TensorSpec> inputs_, outputs_;
  std::unordered_map<std::string, int> in_index_, out_index_;
};

// Named host arrays (the weights files written by dragons_rl/export.py).
struct HostArray {
  TensorSpec spec;
  std::vector<uint8_t> data;
};
std::vector<HostArray> read_arrays(const std::string& path);
void write_arrays(const std::string& path, const std::vector<HostArray>& arrays);

}  // namespace rt
