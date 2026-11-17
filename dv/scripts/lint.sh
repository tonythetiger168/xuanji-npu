#!/bin/bash
# NPU RTL lint: 以 orca_npu_ss_apb 為頂層一次編譯全部 NPU / FP RTL (與 xuanji-core 的 make lint 同一組旗標)
set -euo pipefail
cd "$(dirname "$0")/../.."
verilator --lint-only -Wall -Wno-fatal -Irtl/common -Irtl/fp -f rtl/npu.f --top-module orca_npu_ss_apb
