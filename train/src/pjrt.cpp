#include "pjrt.hpp"

#include <dlfcn.h>

#include <fstream>
#include <sstream>

namespace pjrt {

namespace {
template <typename T>
T Args(size_t size) {
  T a{};
  a.struct_size = size;
  return a;
}
#define ARGS(T) Args<T>(T##_STRUCT_SIZE)

// for teardown paths that must not throw: free an error without reporting it
void Drop(const PJRT_Api* api, PJRT_Error* e) {
  if (!e) return;
  auto d = ARGS(PJRT_Error_Destroy_Args);
  d.error = e;
  api->PJRT_Error_Destroy(&d);
}
}  // namespace

size_t Size(DType t) {
  switch (t) {
    case DType::F32: case DType::S32: case DType::U32: return 4;
    case DType::F16: return 2;
    default: return 1;
  }
}

DType ParseDType(const std::string& s) {
  if (s == "f32") return DType::F32;
  if (s == "f16") return DType::F16;
  if (s == "s32") return DType::S32;
  if (s == "u32") return DType::U32;
  if (s == "u8") return DType::U8;
  if (s == "pred") return DType::Pred;
  throw std::runtime_error("unknown dtype " + s);
}

PJRT_Buffer_Type ToPjrt(DType t) {
  switch (t) {
    case DType::F32: return PJRT_Buffer_Type_F32;
    case DType::F16: return PJRT_Buffer_Type_F16;
    case DType::S32: return PJRT_Buffer_Type_S32;
    case DType::U32: return PJRT_Buffer_Type_U32;
    case DType::U8: return PJRT_Buffer_Type_U8;
    default: return PJRT_Buffer_Type_PRED;
  }
}

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

// ---- runtime ---------------------------------------------------------------------

Runtime::Runtime(const std::string& plugin_path) {
  lib_ = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!lib_) throw std::runtime_error(std::string("cannot load PJRT plugin: ") + dlerror());
  auto get = reinterpret_cast<const PJRT_Api* (*)()>(dlsym(lib_, "GetPjrtApi"));
  if (!get) throw std::runtime_error("the plugin has no GetPjrtApi");
  api_ = get();
  if (api_->pjrt_api_version.major_version != PJRT_API_MAJOR)
    throw std::runtime_error("PJRT major version mismatch between plugin and header");
  auto init = ARGS(PJRT_Plugin_Initialize_Args);
  Check(api_->PJRT_Plugin_Initialize(&init), "initialise plugin");
  auto cc = ARGS(PJRT_Client_Create_Args);
  Check(api_->PJRT_Client_Create(&cc), "create client");
  client_ = cc.client;
  auto ad = ARGS(PJRT_Client_AddressableDevices_Args);
  ad.client = client_;
  Check(api_->PJRT_Client_AddressableDevices(&ad), "list devices");
  devices_.assign(ad.addressable_devices, ad.addressable_devices + ad.num_addressable_devices);
}

Runtime::~Runtime() {
  if (client_) {
    auto d = ARGS(PJRT_Client_Destroy_Args);
    d.client = client_;
    Drop(api_, api_->PJRT_Client_Destroy(&d));
  }
  // the plugin stays loaded: unloading CUDA libraries at exit is not worth the risk
}

void Runtime::Check(PJRT_Error* e, const char* what) const {
  if (!e) return;
  auto m = ARGS(PJRT_Error_Message_Args);
  m.error = e;
  api_->PJRT_Error_Message(&m);
  std::string msg = std::string(what) + ": " + std::string(m.message, m.message_size);
  auto d = ARGS(PJRT_Error_Destroy_Args);
  d.error = e;
  api_->PJRT_Error_Destroy(&d);
  throw std::runtime_error(msg);
}

void Runtime::Await(PJRT_Event* e) const {
  if (!e) return;
  auto a = ARGS(PJRT_Event_Await_Args);
  a.event = e;
  PJRT_Error* err = api_->PJRT_Event_Await(&a);
  auto d = ARGS(PJRT_Event_Destroy_Args);
  d.event = e;
  Drop(api_, api_->PJRT_Event_Destroy(&d));
  Check(err, "await");
}

Buffer Runtime::Upload(int device, const void* data, DType type, const std::vector<int64_t>& dims) const {
  auto b = ARGS(PJRT_Client_BufferFromHostBuffer_Args);
  b.client = client_;
  b.data = data;
  b.type = ToPjrt(type);
  b.dims = dims.data();
  b.num_dims = dims.size();
  b.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  b.device = devices_.at(device);
  Check(api_->PJRT_Client_BufferFromHostBuffer(&b), "upload");
  Await(b.done_with_host_buffer);
  return Buffer(this, b.buffer);
}

// ---- buffers ---------------------------------------------------------------------

Buffer& Buffer::operator=(Buffer&& o) noexcept {
  if (this != &o) {
    Reset();
    rt_ = o.rt_;
    b_ = o.b_;
    o.b_ = nullptr;
  }
  return *this;
}

void Buffer::Reset() {
  if (!b_) return;
  auto d = ARGS(PJRT_Buffer_Destroy_Args);
  d.buffer = b_;
  Drop(rt_->api(), rt_->api()->PJRT_Buffer_Destroy(&d));
  b_ = nullptr;
}

void Buffer::ToHost(void* dst, size_t bytes) const {
  auto t = ARGS(PJRT_Buffer_ToHostBuffer_Args);
  t.src = b_;
  t.dst = dst;
  t.dst_size = bytes;
  rt_->Check(rt_->api()->PJRT_Buffer_ToHostBuffer(&t), "download");
  rt_->Await(t.event);
}

void Buffer::RawToHost(void* dst, int64_t offset, int64_t bytes) const {
  auto t = ARGS(PJRT_Buffer_CopyRawToHost_Args);
  t.buffer = b_;
  t.dst = dst;
  t.offset = offset;
  t.transfer_size = bytes;
  rt_->Check(rt_->api()->PJRT_Buffer_CopyRawToHost(&t), "raw download");
  rt_->Await(t.event);
}

Buffer Buffer::CopyTo(int device) const {
  auto c = ARGS(PJRT_Buffer_CopyToDevice_Args);
  c.buffer = b_;
  c.dst_device = rt_->device(device);
  rt_->Check(rt_->api()->PJRT_Buffer_CopyToDevice(&c), "copy to device");
  return Buffer(rt_, c.dst_buffer);
}

// ---- programs --------------------------------------------------------------------

size_t Arg::bytes() const {
  size_t n = Size(type);
  for (int64_t d : dims) n *= static_cast<size_t>(d);
  return n;
}

namespace {
Signature ReadSignature(const std::string& path) {
  Signature sig;
  std::istringstream in(ReadFile(path));
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream f(line);
    std::string kind, dtype, shape, flag;
    int index;
    f >> kind >> index >> dtype >> shape >> flag;
    Arg a;
    a.type = ParseDType(dtype);
    if (shape != "-") {
      std::istringstream s(shape);
      for (std::string tok; std::getline(s, tok, ',');) a.dims.push_back(std::stoll(tok));
    }
    a.donated = flag == "donated";
    a.dropped = flag == "dropped";
    auto& list = kind == "in" ? sig.in : sig.out;
    if (static_cast<int>(list.size()) != index) throw std::runtime_error("bad signature file " + path);
    list.push_back(a);
  }
  return sig;
}
}  // namespace

Program::Program(const Runtime& rt, const std::string& dir, const std::string& name) : rt_(rt), name_(name) {
  sig_ = ReadSignature(dir + "/" + name + ".sig");
  std::string code = ReadFile(dir + "/" + name + ".mlirbc");
  const std::string opts = ReadFile(dir + "/compile_options.pb");
  auto prog = ARGS(PJRT_Program);
  prog.code = code.data();
  prog.code_size = code.size();
  static const char kFormat[] = "mlir";
  prog.format = kFormat;
  prog.format_size = 4;
  auto c = ARGS(PJRT_Client_Compile_Args);
  c.client = rt.client();
  c.program = &prog;
  c.compile_options = opts.data();
  c.compile_options_size = opts.size();
  rt.Check(rt.api()->PJRT_Client_Compile(&c), ("compile " + name).c_str());
  exe_ = c.executable;
}

Program::~Program() {
  if (!exe_) return;
  auto d = ARGS(PJRT_LoadedExecutable_Destroy_Args);
  d.executable = exe_;
  Drop(rt_.api(), rt_.api()->PJRT_LoadedExecutable_Destroy(&d));
}

std::vector<Buffer> Program::Run(int device, const std::vector<const Buffer*>& args) const {
  if (args.size() != sig_.in.size())
    throw std::runtime_error(name_ + ": expected " + std::to_string(sig_.in.size()) + " arguments, got " +
                             std::to_string(args.size()));
  std::vector<PJRT_Buffer*> in;
  for (size_t i = 0; i < args.size(); ++i) {
    if (sig_.in[i].dropped) continue;
    if (!args[i] || !*args[i]) throw std::runtime_error(name_ + ": argument " + std::to_string(i) + " is empty");
    in.push_back(args[i]->get());
  }
  std::vector<PJRT_Buffer*> out(sig_.out.size(), nullptr);
  PJRT_Buffer* const* arg_lists[1] = {in.data()};
  PJRT_Buffer** out_lists[1] = {out.data()};
  auto opts = ARGS(PJRT_ExecuteOptions);
  auto e = ARGS(PJRT_LoadedExecutable_Execute_Args);
  e.executable = exe_;
  e.options = &opts;
  e.argument_lists = arg_lists;
  e.num_devices = 1;
  e.num_args = in.size();
  e.output_lists = out_lists;
  e.execute_device = rt_.device(device);
  rt_.Check(rt_.api()->PJRT_LoadedExecutable_Execute(&e), ("run " + name_).c_str());
  std::vector<Buffer> result;
  result.reserve(out.size());
  for (PJRT_Buffer* b : out) result.emplace_back(&rt_, b);
  return result;
}

}  // namespace pjrt
