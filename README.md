# xuanji-npu

Xuanji NPU subsystem, split out of [xuanji-core](https://github.com/tonythetiger168/xuanji-core) with its history.

| Path | Contents |
|---|---|
| `rtl/npu/` | NPU core, subsystem (`orca_npu_ss`, APB wrapper), DMA, L2, vector / LLM / MoE engines |
| `rtl/fp/` | Shared floating-point library (also used by xuanji-core's SIS) |
| `rtl/common/` | `npu_pkg`, `npu_vec_pkg`, `clock_gate`, `ret_shadow` |
| `rtl/npu.f` | RTL file list |
| `dv/`, `formal/` | Testbenches, coverage, mutation and formal proofs |
| `accelerators/` | Archived v5 / v6 accelerator generations (Apache-2.0, see `accelerators/PROVENANCE.md`) |

xuanji-core includes this repository as a git submodule at `ip/xuanji-npu`.

Run a flow from the repository root, for example `bash dv/scripts/run_npu_cov.sh nano` (Verilator 5.052).
