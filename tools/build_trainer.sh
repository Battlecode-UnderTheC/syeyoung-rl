#!/usr/bin/env sh
# Builds the native trainer and its self-test against the libtorch that ships
# with the installed torch wheel, then runs the self-test.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-"$root/.venv/bin/python"}
torch_dir=$("$python" -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent)")
abi=$("$python" -c "import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)")
case "$(uname)" in
  Darwin) cuda="" ;;
  *)
    cuda=""
    if ls "$torch_dir/lib" | grep -q libtorch_cuda.so; then
      cuda="-ltorch_cuda -lc10_cuda -DUSE_CUDA"
      # the CUDA runtime headers come with the nvidia pip packages or a toolkit
      for inc in "$torch_dir"/../triton/backends/nvidia/include /usr/local/cuda/include "$torch_dir"/../nvidia/cuda_runtime/include; do
        [ -f "$inc/cuda_runtime_api.h" ] && cuda="$cuda -I$inc" && break
      done
      # and the runtime library itself, for the stream API
      for lib in "$torch_dir"/../nvidia/cu*/lib /usr/local/cuda/lib64; do
        if ls "$lib"/libcudart.so* >/dev/null 2>&1; then
          cudart=$(ls "$lib"/libcudart.so.* | head -1)
          cuda="$cuda $cudart -Wl,-rpath,$lib"
          break
        fi
      done
    fi ;;
esac
common="$root/dragons-env/cpp/dragons/game.cpp $root/cpp/agent/encoder.cpp \
  $root/cpp/agent/spatial/memory.cpp $root/cpp/agent/spatial/codec.cpp $root/cpp/train/gpu_state.cpp \
  $root/cpp/train/model.cpp $root/cpp/train/arena.cpp $root/cpp/train/graphs.cpp $root/cpp/train/mapgen.cpp"
flags="-std=c++20 -O3 -g -pthread -I$root/cpp -I$root/dragons-env/cpp -I$torch_dir/include \
  -I$torch_dir/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=$abi \
  -L$torch_dir/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$torch_dir/lib $cuda"
targets=${TARGETS:-"selftest spatial_train"}
for target in $targets; do
  [ -f "$root/cpp/train/$target.cpp" ] || continue
  # shellcheck disable=SC2086
  c++ -o "$root/cpp/train/$target" "$root/cpp/train/$target.cpp" $common $flags
  echo "built $root/cpp/train/$target"
done
if [ -x "$root/cpp/train/selftest" ]; then "$root/cpp/train/selftest"; fi
