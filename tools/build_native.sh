#!/usr/bin/env sh
# Builds the native sim extension into cpp/.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-"$root/.venv/bin/python"}
inc=$("$python" -m pybind11 --includes)
ext=$("$python" -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
case "$(uname)" in
  Darwin) link="-undefined dynamic_lookup" ;;
  *) link="" ;;
esac
# shellcheck disable=SC2086
c++ -std=c++20 -O3 -shared -fPIC $link $inc -I"$root/cpp" -I"$root/dragons-env/cpp" \
    "$root/dragons-env/cpp/dragons/game.cpp" "$root/cpp/agent/encoder.cpp" \
    "$root/cpp/agent/bindings.cpp" -o "$root/cpp/_dragons$ext"
echo "built $root/cpp/_dragons$ext"
