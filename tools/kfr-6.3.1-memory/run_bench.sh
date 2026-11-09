#!/usr/bin/env bash
# Build bench.cpp against an installed KFR 6.3.1 and collect everything needed to judge the four-step FFT
# on THIS machine: system info, accuracy, a memory/speed matrix against KFR dft_plan, an interleaved A/B
# comparison (robust against machine noise), a block-size sweep and thread scaling.
#
#   ./run_bench.sh --kfr /path/to/kfr-install [--cxx clang++] [--n 200000000] [--reps 5]
#                  [--types "f32 f64"] [--threads 4] [--flags "..."] [--quick]
#
# Output goes to stdout and to $OUT/report.txt (default ./kfr-memory-bench). Send that file back.
set -uo pipefail

KFR=""
CXX=${CXX:-clang++}
N=200000000
REPS=5
TYPES="f32 f64"
THREADS=$(nproc 2>/dev/null || echo 4)
FLAGS="-O3 -DNDEBUG -ffp-contract=fast -pthread"
QUICK=0
while [ $# -gt 0 ]; do
    case "$1" in
        --kfr) KFR=$2; shift 2 ;;
        --cxx) CXX=$2; shift 2 ;;
        --n) N=$2; shift 2 ;;
        --reps) REPS=$2; shift 2 ;;
        --types) TYPES=$2; shift 2 ;;
        --threads) THREADS=$2; shift 2 ;;
        --flags) FLAGS=$2; shift 2 ;;
        --quick) QUICK=1; shift ;;
        -h|--help) sed -n 2,12p "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
[ -n "$KFR" ] || { echo "--kfr is required" >&2; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/kfr-memory-bench}
mkdir -p "$OUT"
exec > >(tee "$OUT/report.txt") 2>&1

echo "== system"
uname -a
lscpu 2>/dev/null | grep -E "Model name|Architecture|^CPU\(s\)|Thread|L1d|L2|L3|NUMA node\(s\)" || true
echo "page size: $(getconf PAGESIZE 2>/dev/null)"
echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
free -g 2>/dev/null | head -2 || true
$CXX --version | head -1

LIBS=$(cd "$KFR/lib" && ls libkfr_dft_*.a | sed 's/^lib\(.*\)\.a$/-l\1/' | tr '\n' ' ')
echo "== build ($CXX $FLAGS)"
$CXX -std=c++17 $FLAGS -I"$KFR/include" "$HERE/bench.cpp" -L"$KFR/lib" \
    -Wl,--start-group $LIBS -Wl,--end-group -o "$OUT/bench" || exit 1
B="$OUT/bench"

echo "== detected L2 cache"
FS_INFO=1 "$B" 1000000 f32 check | head -1

echo "== accuracy and thread checks (every line below must say OK)"
for t in $TYPES; do
    for n in 1000000 2000000 3600000 16777216; do
        "$B" $n $t check | grep -E "^check|OK|FAIL" | grep -v "^  direct" | tr '\n' ' '
        echo
    done
done

echo "== memory / speed, N=$N, $REPS runs each (input regenerated before each run, not timed)"
for t in $TYPES; do
    for m in direct direct_inplace fs_auto fs_inplace; do
        "$B" "$N" $t $m "$REPS" || echo "$m $t failed (out of memory?)"
    done
done

[ "$QUICK" = 1 ] && exit 0

echo "== interleaved A/B (ratio B/A < 1 means B is faster; robust against load changes)"
for t in $TYPES; do
    "$B" "$N" $t ab direct_inplace fs_inplace 8 | tail -4
    "$B" "$N" $t ab direct fs_auto 8 | tail -4
done

echo "== autotune: time the most promising N1 x N2 splits on the real size (keep the best as options.n1)"
for t in $TYPES; do
    "$B" "$N" $t tune inplace
    "$B" "$N" $t tune oop
done

echo "== block size sweep (FS_B), single thread, fs_inplace"
for t in $TYPES; do
    for b in 8 16 32 64; do
        echo -n "B=$b  "
        FS_B=$b "$B" "$N" $t fs_inplace 3 | sed 's/plan=[^ ]* //'
    done
done

echo "== thread scaling (opt-in multi-threading), fs_inplace"
for t in $TYPES; do
    for th in 2 4 8 16; do
        [ "$th" -le "$THREADS" ] || continue
        "$B" "$N" $t ab fs_inplace fs_inplace@$th 5 | tail -4
    done
done
