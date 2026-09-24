// The runtime layer over the PJRT C API: devices, typed buffers, and programs
// whose arguments and results are addressed by name.
//
// No training concepts here. Everything above talks to programs through the
// typed wrappers in train/programs.hpp.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace pjrt {

struct Device {
  int index;
  friend bool operator==(Device, Device) = default;
};

enum class DType : uint8_t { F32, F16, S32, U32, U8, Pred };
template <typename T> constexpr DType DTypeOf();   // float -> F32, int32_t -> S32, uint8_t -> U8, ...

struct Shape {
  std::vector<int64_t> dims;
  int64_t elements() const;
  friend bool operator==(const Shape&, const Shape&) = default;
};

class Runtime;

// One device buffer; move-only; destroyed with its owner.
class Buffer {
 public:
  Buffer(Buffer&&) noexcept;
  Buffer& operator=(Buffer&&) noexcept;
  ~Buffer();

  Device device() const;
  DType dtype() const;
  const Shape& shape() const;

  // Copies the whole buffer into `out` (sizes and types are checked).
  template <typename T> void Read(std::span<T> out) const;
  // Copies rows [0, rows) of the leading dimension.
  template <typename T> void ReadRows(int64_t rows, std::span<T> out) const;
  Buffer CopyTo(Device d) const;

 private:
  friend class Runtime;
  friend class Program;
  struct Handle;
  std::unique_ptr<Handle> h_;
};

struct RuntimeOptions {
  std::string plugin_path;           // e.g. .../jax_plugins/xla_cuda13/xla_cuda_plugin.so
  float memory_fraction = 0.9f;      // of each GPU the XLA allocator may take
};

class Runtime {
 public:
  explicit Runtime(const RuntimeOptions& options);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  int device_count() const;
  Device device(int index) const;

  template <typename T> Buffer Upload(Device d, std::span<const T> data, Shape shape) const;
  Buffer Zeros(Device d, DType type, Shape shape) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A compiled program and its named signature.
struct ArgSpec {
  std::string name;
  DType type;
  Shape shape;
  bool donated;                      // consumed by a run (the caller hands over ownership)
};

class Program {
 public:
  // Compiles <dir>/<name>.mlirbc; the signature comes from <dir>/<name>.sig.
  Program(const Runtime& rt, const std::string& dir, const std::string& name);
  ~Program();

  const std::string& name() const;
  std::span<const ArgSpec> inputs() const;
  std::span<const ArgSpec> outputs() const;

  // Arguments by name. A donated input must be passed as an owned Buffer (it
  // is consumed); a borrowed one as a reference. Every input must be given,
  // with the declared type and shape, or Run throws before launching.
  class Args {
   public:
    Args& Borrow(const std::string& name, const Buffer& b);
    Args& Give(const std::string& name, Buffer&& b);
    Args& BorrowAll(const std::string& prefix, std::span<const Buffer> leaves);   // a pytree's leaves, in order
    Args& GiveAll(const std::string& prefix, std::vector<Buffer>&& leaves);
  };
  // Results by name.
  class Results {
   public:
    Buffer Take(const std::string& name);
    std::vector<Buffer> TakeAll(const std::string& prefix);
  };

  Results Run(Device d, Args&& args) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pjrt
