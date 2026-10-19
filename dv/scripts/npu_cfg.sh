# NPU 產品組態 (與 rtl/ai/npu_products.sv 一致): source 後呼叫 npu_cfg <name> 設定 N NCU NB DEPTH MIN_UTIL
# MIN_UTIL: 效能回歸門檻 (大 K 的 MAC 利用率 %, 低於即失敗); 量測值見 docs/npu.md
npu_cfg() {
  case "$1" in
    nano) N=4;  NCU=1; NB=2;  DEPTH=512;  MIN_UTIL=95 ;;
    x1)   N=8;  NCU=4; NB=4;  DEPTH=1024; MIN_UTIL=95 ;;
    pro)  N=16; NCU=8; NB=16; DEPTH=1024; MIN_UTIL=95 ;;
    *) echo "unknown NPU config '$1' (nano / x1 / pro)" >&2; return 1 ;;
  esac
}
