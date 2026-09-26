#include "pjrt.hpp"

#include <dlfcn.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rt {

size_t dtype_size(DType t) {
  switch (t) {
    case DType::F32: case DType::S32: return 4;
    case DType::F16: return 2;
    default: return 1;
  }
}

size_t TensorSpec::bytes() const {
  size_t n = dtype_size(type);
  for (const int64_t d : dims) n *= static_cast<size_t>(d);
  return n;
}

namespace {

DType parse_dtype(const std::string& t) {
  if (t == "f32") return DType::F32;
  if (t == "s32") return DType::S32;
  if (t == "u8") return DType::U8;
  if (t == "f16") return DType::F16;
  if (t == "pred") return DType::PRED;
  throw std::invalid_argument("unknown dtype " + t);
}

PJRT_Buffer_Type pjrt_type(DType t) {
  switch (t) {
    case DType::F32: return PJRT_Buffer_Type_F32;
    case DType::S32: return PJRT_Buffer_Type_S32;
    case DType::U8: return PJRT_Buffer_Type_U8;
    case DType::F16: return PJRT_Buffer_Type_F16;
    default: return PJRT_Buffer_Type_PRED;
  }
}

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

}  // namespace

// ---- runtime ----------------------------------------------------------------------------------

Runtime::Runtime(const std::string& plugin_path) {
  handle_ = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle_) throw std::runtime_error(std::string("dlopen: ") + dlerror());
  auto get = reinterpret_cast<const PJRT_Api* (*)()>(dlsym(handle_, "GetPjrtApi"));
  if (!get) throw std::runtime_error("no GetPjrtApi in " + plugin_path);
  api_ = get();
  PJRT_Plugin_Initialize_Args ia{};
  ia.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  check(api_->PJRT_Plugin_Initialize(&ia), "plugin initialize");
  PJRT_Client_Create_Args ca{};
  ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  check(api_->PJRT_Client_Create(&ca), "client create");
  client_ = ca.client;
  PJRT_Client_AddressableDevices_Args da{};
  da.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  da.client = client_;
  check(api_->PJRT_Client_AddressableDevices(&da), "devices");
  devices_.assign(da.addressable_devices, da.addressable_devices + da.num_addressable_devices);
}

Runtime::~Runtime() {
  if (client_) {
    PJRT_Client_Destroy_Args d{};
    d.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    d.client = client_;
    api_->PJRT_Client_Destroy(&d);
  }
}

void Runtime::check(PJRT_Error* e, const char* what) const {
  if (!e) return;
  PJRT_Error_Message_Args m{};
  m.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  m.error = e;
  api_->PJRT_Error_Message(&m);
  std::string msg(m.message, m.message_size);
  PJRT_Error_Destroy_Args d{};
  d.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  d.error = e;
  api_->PJRT_Error_Destroy(&d);
  throw std::runtime_error(std::string(what) + ": " + msg);
}

void Runtime::await(PJRT_Event* e) const {
  if (!e) return;
  PJRT_Event_Await_Args a{};
  a.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  a.event = e;
  PJRT_Error* err = api_->PJRT_Event_Await(&a);
  PJRT_Event_Destroy_Args d{};
  d.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  d.event = e;
  api_->PJRT_Event_Destroy(&d);
  check(err, "await");
}

// ---- buffers -----------------------------------------------------------------------------------

Buffer& Buffer::operator=(Buffer&& o) noexcept {
  if (this != &o) {
    this->~Buffer();
    rt_ = o.rt_;
    b_ = o.b_;
    o.b_ = nullptr;
  }
  return *this;
}

Buffer::~Buffer() {
  if (b_) {
    PJRT_Buffer_Destroy_Args d{};
    d.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    d.buffer = b_;
    rt_->api()->PJRT_Buffer_Destroy(&d);
    b_ = nullptr;
  }
}

void Buffer::download(void* dst, size_t dst_bytes) const {
  PJRT_Buffer_ToHostBuffer_Args t{};
  t.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  t.src = b_;
  t.dst = dst;
  t.dst_size = dst_bytes;
  rt_->check(rt_->api()->PJRT_Buffer_ToHostBuffer(&t), "download");
  rt_->await(t.event);
}

Buffer upload(const Runtime& rt, int device, const TensorSpec& spec, const void* data) {
  PJRT_Client_BufferFromHostBuffer_Args b{};
  b.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  b.client = rt.client();
  b.data = data;
  b.type = pjrt_type(spec.type);
  b.dims = spec.dims.data();
  b.num_dims = spec.dims.size();
  b.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  b.device = rt.device(device);
  rt.check(rt.api()->PJRT_Client_BufferFromHostBuffer(&b), "upload");
  rt.await(b.done_with_host_buffer);
  return Buffer(&rt, b.buffer);
}

Transfers::~Transfers() {
  try {
    wait();
  } catch (...) {
  }
}

Buffer Transfers::upload(int device, const TensorSpec& spec, const void* data) {
  PJRT_Client_BufferFromHostBuffer_Args b{};
  b.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  b.client = rt_.client();
  b.data = data;
  b.type = pjrt_type(spec.type);
  b.dims = spec.dims.data();
  b.num_dims = spec.dims.size();
  b.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  b.device = rt_.device(device);
  rt_.check(rt_.api()->PJRT_Client_BufferFromHostBuffer(&b), "upload");
  if (b.done_with_host_buffer) events_.push_back(b.done_with_host_buffer);
  return Buffer(&rt_, b.buffer);
}

void Transfers::download(const Buffer& src, void* dst, size_t bytes) {
  PJRT_Buffer_ToHostBuffer_Args t{};
  t.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  t.src = src.get();
  t.dst = dst;
  t.dst_size = bytes;
  rt_.check(rt_.api()->PJRT_Buffer_ToHostBuffer(&t), "download");
  if (t.event) events_.push_back(t.event);
}

void Transfers::wait() {
  std::vector<PJRT_Event*> events;
  events.swap(events_);
  for (PJRT_Event* e : events) rt_.await(e);
}

// ---- programs ----------------------------------------------------------------------------------

Program::Program(const Runtime& rt, const std::string& dir, const std::string& name, int device)
    : rt_(rt), device_(device) {
  std::istringstream m(slurp(dir + "/" + name + ".txt"));
  std::string kind, tname, t;
  int rank;
  while (m >> kind >> tname >> t >> rank) {
    TensorSpec s{tname, parse_dtype(t), std::vector<int64_t>(rank)};
    for (auto& d : s.dims) m >> d;
    if (kind == "in") {
      in_index_[tname] = static_cast<int>(inputs_.size());
      inputs_.push_back(std::move(s));
    } else {
      out_index_[tname] = static_cast<int>(outputs_.size());
      outputs_.push_back(std::move(s));
    }
  }
  const std::string code = slurp(dir + "/" + name + ".mlirbc");
  const std::string opts = slurp(dir + "/" + name + ".options." + std::to_string(device));
  PJRT_Program prog{};
  prog.struct_size = PJRT_Program_STRUCT_SIZE;
  prog.code = const_cast<char*>(code.data());
  prog.code_size = code.size();
  prog.format = "mlir";
  prog.format_size = 4;
  PJRT_Client_Compile_Args cc{};
  cc.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  cc.client = rt.client();
  cc.program = &prog;
  cc.compile_options = opts.data();
  cc.compile_options_size = opts.size();
  rt.check(rt.api()->PJRT_Client_Compile(&cc), ("compile " + name).c_str());
  exe_ = cc.executable;
}

Program::~Program() {
  if (exe_) {
    PJRT_LoadedExecutable_Destroy_Args d{};
    d.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    d.executable = exe_;
    rt_.api()->PJRT_LoadedExecutable_Destroy(&d);
  }
}

int Program::input(const std::string& name) const {
  auto it = in_index_.find(name);
  if (it == in_index_.end()) throw std::invalid_argument("no program input " + name);
  return it->second;
}

int Program::output(const std::string& name) const {
  auto it = out_index_.find(name);
  if (it == out_index_.end()) throw std::invalid_argument("no program output " + name);
  return it->second;
}

std::vector<Buffer> Program::run(const std::vector<PJRT_Buffer*>& args) const {
  if (args.size() != inputs_.size()) throw std::invalid_argument("argument count");
  std::vector<PJRT_Buffer*> outs(outputs_.size(), nullptr);
  PJRT_Buffer* const* arg_list = args.data();
  PJRT_Buffer** out_list = outs.data();
  PJRT_ExecuteOptions eo{};
  eo.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  PJRT_LoadedExecutable_Execute_Args ex{};
  ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  ex.executable = exe_;
  ex.options = &eo;
  ex.argument_lists = &arg_list;
  ex.num_devices = 1;
  ex.num_args = args.size();
  ex.output_lists = &out_list;
  rt_.check(rt_.api()->PJRT_LoadedExecutable_Execute(&ex), "execute");   // async: downloads wait for the results
  std::vector<Buffer> result;
  result.reserve(outs.size());
  for (PJRT_Buffer* b : outs) result.emplace_back(&rt_, b);
  return result;
}

// ---- named arrays --------------------------------------------------------------------------------

std::vector<HostArray> read_arrays(const std::string& path) {
  const std::string s = slurp(path);
  size_t at = 0;
  auto take = [&](void* dst, size_t n) {
    if (at + n > s.size()) throw std::runtime_error("truncated arrays file " + path);
    std::memcpy(dst, s.data() + at, n);
    at += n;
  };
  uint32_t count;
  take(&count, 4);
  std::vector<HostArray> out(count);
  for (auto& a : out) {
    uint32_t len;
    take(&len, 4);
    a.spec.name.resize(len);
    take(a.spec.name.data(), len);
    uint8_t code;
    uint32_t rank;
    take(&code, 1);
    take(&rank, 4);
    a.spec.type = static_cast<DType>(code);
    a.spec.dims.resize(rank);
    take(a.spec.dims.data(), rank * 8);
    a.data.resize(a.spec.bytes());
    take(a.data.data(), a.data.size());
  }
  return out;
}

void write_arrays(const std::string& path, const std::vector<HostArray>& arrays) {
  std::ofstream f(path + ".tmp", std::ios::binary);
  const uint32_t count = static_cast<uint32_t>(arrays.size());
  f.write(reinterpret_cast<const char*>(&count), 4);
  for (const auto& a : arrays) {
    const uint32_t len = static_cast<uint32_t>(a.spec.name.size());
    f.write(reinterpret_cast<const char*>(&len), 4);
    f.write(a.spec.name.data(), len);
    const uint8_t code = static_cast<uint8_t>(a.spec.type);
    const uint32_t rank = static_cast<uint32_t>(a.spec.dims.size());
    f.write(reinterpret_cast<const char*>(&code), 1);
    f.write(reinterpret_cast<const char*>(&rank), 4);
    f.write(reinterpret_cast<const char*>(a.spec.dims.data()), rank * 8);
    f.write(reinterpret_cast<const char*>(a.data.data()), a.data.size());
  }
  f.close();
  if (std::rename((path + ".tmp").c_str(), path.c_str()) != 0) throw std::runtime_error("cannot write " + path);
}

}  // namespace rt
