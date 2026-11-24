#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SHAPE="${1:?用法：run_sanitizer.sh standalone|operator 模式 [--repaired]}"
MODE="${2:?请选择 memcheck、racecheck、initcheck、synccheck 或 baseline}"
REPAIRED="${3:-}"
[[ "$SHAPE" == standalone || "$SHAPE" == operator ]]
[[ -z "$REPAIRED" || "$REPAIRED" == --repaired ]]

signature_for() {
  case "$1" in
    memcheck)  echo 'illegal (read|write)|out[- ]of[- ]bounds' ;;
    racecheck) echo 'RAW hazard|data race' ;;
    initcheck) echo 'uninitialized' ;;
    synccheck) echo 'unpaired[ _-]*set_flag|unpaired set_flag' ;;
    *) return 2 ;;
  esac
}

run_expected() {
  local binary="$1" log="$2" rc status
  echo "执行：mssanitizer --tool=$MODE $binary"
  set +e
  mssanitizer --tool="$MODE" "$binary" 2>&1 | tee "$log"
  rc="${PIPESTATUS[0]}"
  set -e
  if [[ "$REPAIRED" == --repaired ]]; then
    [[ "$rc" -eq 0 ]]
    if grep -Eiq 'uninitialized|illegal (read|write)|RAW hazard|unpaired[ _-]*set_flag' "$log"; then
      echo '修复后仍有诊断，请查看日志' >&2
      return 1
    fi
    grep -iq 'test pass' "$log"
    status=PASS
  else
    if ! grep -Eiq "$(signature_for "$MODE")" "$log"; then
      echo "SANITIZER_RESULT status=FAIL mode=$MODE exit_code=$rc reason=missing_signature" >&2
      return 1
    fi
    status=EXPECTED_DIAGNOSTIC
  fi
  echo "SANITIZER_RESULT status=$status shape=$SHAPE mode=$MODE exit_code=$rc log=$log"
}

FAULT_MODE="$MODE"
SANITIZER=ON
if [[ "$MODE" == baseline ]]; then
  [[ "$SHAPE" == standalone && -z "$REPAIRED" ]]
  SANITIZER=OFF
else
  signature_for "$MODE" > /dev/null
fi
if [[ "$REPAIRED" == --repaired ]]; then
  FAULT_MODE=baseline
fi

if [[ "$SHAPE" == standalone ]]; then
  BUILD="build/standalone-$MODE"
  if [[ "$REPAIRED" == --repaired ]]; then BUILD="build/repaired-$MODE"; fi
  cmake -S src -B "$BUILD" -DLAB05_FAULT_MODE="$FAULT_MODE" -DLAB05_ENABLE_SANITIZER="$SANITIZER"
  cmake --build "$BUILD" -j
  if [[ "$MODE" == baseline ]]; then
    "$BUILD/lab05_add_sanitizer"
    echo 'BASELINE_STATUS=PASS'
  else
    run_expected "$BUILD/lab05_add_sanitizer" "$BUILD/mssanitizer.log"
  fi
else
  ROOT="$PWD/src/operator_project_sanitizer"
  BUILD="$ROOT/build-$MODE"
  if [[ "$REPAIRED" == --repaired ]]; then
    BUILD="$ROOT/build-repaired-$MODE"
  fi
  INSTALL="$(mktemp -d "${TMPDIR:-/tmp}/cann-debug-operator.XXXXXX")"
  HOST="$INSTALL/host-build"
  trap 'chmod -R u+rwX -- "$INSTALL" 2>/dev/null || true; rm -rf -- "$INSTALL"' EXIT
  cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DLAB05_OPERATOR_FAULT_MODE="$FAULT_MODE" -DLAB05_ENABLE_SANITIZER=ON
  cmake --build "$BUILD" --target binary package -j
  PACKAGE=("$BUILD"/custom_opp_*.run)
  [[ -f "${PACKAGE[0]}" ]] || { echo 'operator package missing' >&2; exit 1; }
  "${PACKAGE[0]}" --install-path="$INSTALL"
  cmake -S "$ROOT/test" -B "$HOST" -DCMAKE_SKIP_RPATH=TRUE -DLAB05_CUSTOM_OPP_ROOT="$INSTALL"
  cmake --build "$HOST" -j
  export ASCEND_CUSTOM_OPP_PATH="$INSTALL/vendors/customize"
  export LD_LIBRARY_PATH="$INSTALL/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"
  run_expected "$HOST/execute_add_msanitizer" "$BUILD/mssanitizer.log"
fi
