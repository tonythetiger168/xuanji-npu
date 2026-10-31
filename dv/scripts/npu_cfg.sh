# 建置一律 --output-groups 64: Verilator 預設把產生的 C++ 併成「建置 job 數」個編譯單元, Max (538 MB C++) 時每個單元
# 數十 GB 記憶體 (本機 15 GB 與 CI runner 都被 OOM 終止); 64 組時 -j 4 峰值約 9.4 GB
# NPU 產品組態 (與 rtl/npu/npu_products.sv 一致): source 後呼叫 npu_cfg <name> 設定 N NCU NB DEPTH MIN_UTIL
# TILE_OUT: NPU Subsystem DMA 在途上限 (外部讀取延遲 L 時吞吐量 = min(1, TILE_OUT / (L + 3)) line/拍)
# MIN_UTIL: 效能回歸門檻 (大 K 的 MAC 利用率 %, 低於即失敗); 量測值見 docs/npu.md
npu_cfg() {
  case "$1" in
    nano) N=4;  NCU=1; NB=2;  DEPTH=512;  MIN_UTIL=95; TILE_OUT=16 ;;
    x1)   N=8;  NCU=4; NB=4;  DEPTH=1024; MIN_UTIL=95; TILE_OUT=32 ;;
    pro)  N=16; NCU=8; NB=16; DEPTH=1024; MIN_UTIL=95; TILE_OUT=64 ;;
    max)  N=32; NCU=24; NB=32; DEPTH=1024; MIN_UTIL=78; TILE_OUT=128 ;;   # K 上限 958 (24 個命令同時放進 2 MiB L2)
    *) echo "unknown NPU config '$1' (nano / x1 / pro / max)" >&2; return 1 ;;
  esac
}
