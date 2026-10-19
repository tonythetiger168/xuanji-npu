#!/bin/bash
# 由原始碼安裝指定版本的 Verilator 到 $PREFIX (預設 ~/verilator-<ver>), 已存在則跳過 (配合 actions/cache)。
# 用法: ci/install_verilator.sh [版本=v5.052]; 之後把 $PREFIX/bin 加進 PATH
set -euo pipefail
VER=${1:-v5.052}
PREFIX=${PREFIX:-$HOME/verilator-$VER}
if [ -x "$PREFIX/bin/verilator" ]; then "$PREFIX/bin/verilator" --version; exit 0; fi
sudo apt-get update
sudo apt-get install -y git autoconf g++ flex bison libfl-dev help2man make perl python3 ccache
SRC=$(mktemp -d)
git clone --depth 1 --branch "$VER" https://github.com/verilator/verilator "$SRC"
cd "$SRC" && autoconf && ./configure --prefix="$PREFIX" && make -j"$(nproc)" && make install
"$PREFIX/bin/verilator" --version
