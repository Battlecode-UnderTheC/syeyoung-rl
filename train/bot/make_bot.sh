#!/bin/sh
# Assembles the submission bot (a self-contained unswbc C++ project) from the
# sources here and a trainer checkpoint's actor weights.
#   bot/make_bot.sh <actor.bin> <out dir>
set -e
weights=$1; out=$2
here=$(cd "$(dirname "$0")/.." && pwd)
engine="$here/../dragons-env/cpp/dragons"
rm -rf "$out" && mkdir -p "$out"
cp "$here/bot/main.cpp" "$out/"
for f in observation facts collector critic policy; do cp "$here/native/$f.hpp" "$here/native/$f.cpp" "$out/"; done
cp "$engine"/*.hpp "$engine"/*.cpp "$out/"
# one flat directory (the toolkit compiles without include paths): "dragons/x.hpp" -> "x.hpp"
sed -i.bak 's#"dragons/\([a-z_]*\.hpp\)"#"\1"#' "$out"/*.hpp "$out"/*.cpp && rm -f "$out"/*.bak
cp "$weights" "$out/weights.bin"
cat > "$out/bot.toml" <<TOML
# only translation units belong in include; headers are pulled in by #include
[project]
language = "cpp"
include = ["*.cpp"]
TOML
echo "bot in $out ($(du -sh "$out" | cut -f1))"
