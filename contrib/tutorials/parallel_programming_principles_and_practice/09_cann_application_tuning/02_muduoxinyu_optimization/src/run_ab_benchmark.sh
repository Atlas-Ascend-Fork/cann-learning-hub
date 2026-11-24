#!/usr/bin/env bash
# Run the fixed MuduoXinyu implementation-package A/B experiment.
set -euo pipefail

usage() {
  echo "Usage: bash run_ab_benchmark.sh --muduo-root <dir> --model <file> --tokenizer <file> --output-dir <dir> [--build-log <file>]" >&2
  exit 2
}

MUDUO_ROOT=""
MODEL=""
TOKENIZER=""
OUTPUT=""
BUILD_LOG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --muduo-root) MUDUO_ROOT="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --tokenizer) TOKENIZER="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT="${2:-}"; shift 2 ;;
    --build-log) BUILD_LOG="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$MUDUO_ROOT" && -n "$MODEL" && -n "$TOKENIZER" && -n "$OUTPUT" ]] || usage

mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
# 重跑时先移除旧结论，运行失败后不会读到上次的 PASS。
rm -f -- "$OUTPUT/result.json"

MUDUO_ROOT="$(readlink -f -- "$MUDUO_ROOT")"
MODEL="$(readlink -f -- "$MODEL")"
TOKENIZER="$(readlink -f -- "$TOKENIZER")"
BINARY="$MUDUO_ROOT/muduoXinyu"
for file in "$BINARY" "$MODEL" "$TOKENIZER"; do
  [[ -f "$file" && -s "$file" ]] || { echo "找不到文件或文件为空：$file" >&2; exit 2; }
done
[[ -x "$BINARY" ]] || { echo "程序不可执行：$BINARY" >&2; exit 2; }
if [[ -n "$BUILD_LOG" && ! -f "$BUILD_LOG" ]]; then
  echo "找不到构建日志：$BUILD_LOG" >&2
  exit 2
fi

ASCEND_PATH="${ASCEND_HOME:-${ASCEND_HOME_PATH:-${ASCEND_PATH:-}}}"
: "${ASCEND_PATH:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}"
source "$ASCEND_PATH/set_env.sh"

SMOKE_STEPS=12
PERF_STEPS=120
TIMEOUT_SECONDS=600
PROMPTS="$OUTPUT/prompts_4.txt"
printf 'hello\nhello\nhello\nhello\n' > "$PROMPTS"

# A/B 固定模型、输入和解码参数，但实现包不同：Path A=FP32 多算子，Path B=FP16 FlashAttentionV4。
COMMON=(--backend npu --enableDeviceOpt --skipValidation
        --temperature 0 --topP 1 --dumpTokenSummary)

COMMAND_LOG="$OUTPUT/commands.log"
EXIT_LOG="$OUTPUT/exit_codes.log"
: > "$COMMAND_LOG"
: > "$EXIT_LOG"

run_case() {
  local label="$1" logfile="$2"
  shift 2
  printf '%s\t' "$label" >> "$COMMAND_LOG"
  printf '%q ' timeout "$TIMEOUT_SECONDS" "$@" >> "$COMMAND_LOG"
  printf '\n' >> "$COMMAND_LOG"
  set +e
  timeout "$TIMEOUT_SECONDS" "$@" > "$logfile" 2>&1
  local rc=$?
  set -e
  printf '%s=%s\n' "$label" "$rc" >> "$EXIT_LOG"
  if [[ "$rc" -ne 0 ]]; then
    echo "RUN_CASE=FAIL label=$label exit_code=$rc log=$logfile" >&2
    return "$rc"
  fi
  echo "RUN_CASE=PASS label=$label exit_code=0 log=$logfile"
}

if [[ -n "$BUILD_LOG" ]]; then cp "$BUILD_LOG" "$OUTPUT/build.log"; fi

# 冒烟顺序 A→B；带 profiling 检查路径标记、调用次数、fallback 和 token exact。
run_case smoke_a "$OUTPUT/smoke_path_a.log" \
  "$BINARY" "$MODEL" "$TOKENIZER" hello \
  "${COMMON[@]}" --steps "$SMOKE_STEPS" \
  --profileNpu --profileNpuStart 0 --profileNpuEnd "$SMOKE_STEPS"
run_case smoke_b "$OUTPUT/smoke_path_b.log" \
  "$BINARY" "$MODEL" "$TOKENIZER" hello \
  "${COMMON[@]}" --steps "$SMOKE_STEPS" --useNpuFlashAttention \
  --profileNpu --profileNpuStart 0 --profileNpuEnd "$SMOKE_STEPS"

# 性能顺序反转为 B→A，减小固定运行顺序偏差；sample 1 预热，sample 2~4 统计。
run_case perf_b "$OUTPUT/perf_path_b.log" \
  "$BINARY" "$MODEL" "$TOKENIZER" "$PROMPTS" \
  "${COMMON[@]}" --steps "$PERF_STEPS" --useNpuFlashAttention
run_case perf_a "$OUTPUT/perf_path_a.log" \
  "$BINARY" "$MODEL" "$TOKENIZER" "$PROMPTS" \
  "${COMMON[@]}" --steps "$PERF_STEPS"

python3 "$(dirname "$0")/analyze_results.py" \
  --smoke-a "$OUTPUT/smoke_path_a.log" \
  --smoke-b "$OUTPUT/smoke_path_b.log" \
  --perf-a "$OUTPUT/perf_path_a.log" \
  --perf-b "$OUTPUT/perf_path_b.log" \
  --output "$OUTPUT/result.json" \
  --smoke-steps "$SMOKE_STEPS" --num-layers 12

echo "RUN_AB=PASS result=$OUTPUT/result.json"
