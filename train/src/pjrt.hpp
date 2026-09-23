// A thin C++ layer over the PJRT C API: load a PJRT plugin (the CUDA one that
// ships with jax), compile the programs dragons_model/export.py wrote, move
// buffers between host and devices, and run.
//
// Everything here is synchronous from the caller's point of view except
// Program::Run, which returns as soon as the work is queued; reading an output
// back (Buffer::ToHost) waits for it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pjrt_c_api.h"

namespace pjrt {

enum class DType { F32, F16, S32, U32, U8, Pred };
size_t Size(DType t);
DType ParseDType(const std::string& s);
PJRT_Buffer_Type ToPjrt(DType t);

class Runtime;

// Owns one device buffer.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Runtime* rt, PJRT_Buffer* b) : rt_(rt), b_(b) {}
  Buffer(Buffer&& o) noexcept : rt_(o.rt_), b_(o.b_) { o.b_ = nullptr; }
  Buffer& operator=(Buffer&& o) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { Reset(); }

  PJRT_Buffer* get() const { return b_; }
  explicit operator bool() const { return b_ != nullptr; }
  void Reset();
  // whole buffer, or `bytes` from `offset` of its raw contents
  void ToHost(void* dst, size_t bytes) const;
  void RawToHost(void* dst, int64_t offset, int64_t bytes) const;
  Buffer CopyTo(int device) const;

 private:
  const Runtime* rt_ = nullptr;
  PJRT_Buffer* b_ = nullptr;
};

class Runtime {
 public:
  explicit Runtime(const std::string& plugin_path);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  const PJRT_Api* api() const { return api_; }
  PJRT_Client* client() const { return client_; }
  int num_devices() const { return static_cast<int>(devices_.size()); }
  PJRT_Device* device(int i) const { return devices_.at(i); }

  // A new device buffer holding a copy of host `data` (the host memory may be
  // reused as soon as this returns).
  Buffer Upload(int device, const void* data, DType type, const std::vector<int64_t>& dims) const;
  void Check(PJRT_Error* e, const char* what) const;
  void Await(PJRT_Event* e) const;

 private:
  void* lib_ = nullptr;
  const PJRT_Api* api_ = nullptr;
  PJRT_Client* client_ = nullptr;
  std::vector<PJRT_Device*> devices_;
};

// One flat argument or result of a program, as export.py's .sig lists it.
struct Arg {
  DType type;
  std::vector<int64_t> dims;
  bool donated = false;
  bool dropped = false;     // unused by the program: not passed to the compiled module
  size_t bytes() const;
};
struct Signature {
  std::vector<Arg> in, out;
};

class Program {
 public:
  // Compiles <dir>/<name>.mlirbc with <dir>/compile_options.pb.
  Program(const Runtime& rt, const std::string& dir, const std::string& name);
  ~Program();
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  const Signature& signature() const { return sig_; }
  const std::string& name() const { return name_; }
  // Queues a run on `device`. Donated inputs are consumed: their Buffers must
  // not be used again (the caller replaces them with the matching outputs).
  std::vector<Buffer> Run(int device, const std::vector<const Buffer*>& args) const;

 private:
  const Runtime& rt_;
  std::string name_;
  Signature sig_;
  PJRT_LoadedExecutable* exe_ = nullptr;
};

std::string ReadFile(const std::string& path);

}  // namespace pjrt
