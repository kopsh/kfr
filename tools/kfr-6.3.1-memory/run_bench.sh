#!/usr/bin/env bash
# Build bench.cpp against an installed KFR 6.3.1 and run the accuracy check plus the
# memory/speed matrix (KFR dft_plan vs four-step) on this machine.
#
#   ./run_bench.sh --kfr /path/to/kfr-install [--cxx clang++] [--n 200000000] [--reps 5]
#                  [--types "f32 f64"] [--flags "..."]
set -euo pipefail

KFR=""
CXX=${CXX:-clang++}
N=200000000
REPS=5
TYPES="f32 f64"
FLAGS="-O3 -DNDEBUG -ffp-contract=fast"
while [ $# -gt 0 ]; do
    case "$1" in
        --kfr) KFR=$2; shift 2 ;;
        --cxx) CXX=$2; shift 2 ;;
        --n) N=$2; shift 2 ;;
        --reps) REPS=$2; shift 2 ;;
        --types) TYPES=$2; shift 2 ;;
        --flags) FLAGS=$2; shift 2 ;;
        -h|--help) sed -n 2,7p "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
[ -n "$KFR" ] || { echo "--kfr is required" >&2; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/kfr-memory-bench}
mkdir -p "$OUT"
LIBS=$(cd "$KFR/lib" && ls libkfr_dft_*.a | sed 's/^lib\(.*\)\.a$/-l\1/' | tr '\n' ' ')
echo "== build ($CXX $FLAGS, $LIBS)"
$CXX -std=c++17 $FLAGS -I"$KFR/include" "$HERE/bench.cpp" -L"$KFR/lib" \
    -Wl,--start-group $LIBS -Wl,--end-group -o "$OUT/bench"

echo "== accuracy"
for t in $TYPES; do
    for n in 1000000 2000000 16777216; do "$OUT/bench" $n $t check; done
done

echo "== memory / speed, N=$N, $REPS runs each (input regenerated before each run, not timed)"
free -g 2>/dev/null | head -2 || true
for t in $TYPES; do
    for m in direct direct_inplace fs_destroy fs_preserve fs_inplace; do
        "$OUT/bench" "$N" $t $m "$REPS" || echo "$m $t failed (out of memory?)"
    done
done | tee "$OUT/matrix.txt"
