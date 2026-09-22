#!/usr/bin/env sh
# Builds the native rollout driver against the libtorch that ships with the
# installed torch wheel.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-"$root/.venv/bin/python"}
torch_dir=$("$python" -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent)")
abi=$("$python" -c "import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)")
for target in rollout rollout_mt train; do
c++ -std=c++20 -O3 -pthread -o "$root/cpp/train/$target" \
    "$root/cpp/train/$target.cpp" "$root/dragons-env/cpp/dragons/game.cpp" "$root/cpp/agent/encoder.cpp" \
    -I"$root/cpp" -I"$root/dragons-env/cpp" -I"$torch_dir/include" \
    -I"$torch_dir/include/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI="$abi" \
    -L"$torch_dir/lib" -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$torch_dir/lib" \
    $(ls "$torch_dir/lib" | grep -q libtorch_cuda.so && echo "-ltorch_cuda -lc10_cuda")
echo "built $root/cpp/train/$target"
done
