#!/usr/bin/env bash
# Builds KFR 6.3.1 with Clang against a GCC 10 toolchain (libstdc++ headers and runtime), builds an
# example library with a C interface on top of it, links it into an application compiled with GCC 10,
# and checks that nothing requires libstdc++ symbols newer than the GCC 10 runtime provides.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage()
{
    cat <<EOF
Usage: $0 --kfr-src DIR (--gcc-toolchain DIR | --gcc-install-dir DIR) [options]

Required:
  --kfr-src DIR          KFR 6.3.1 source tree
  --gcc-toolchain DIR    GCC 10 installation prefix, passed to clang as --gcc-toolchain (clang >= 12).
                         Clang picks the newest GCC under the prefix, so the prefix must contain only GCC 10
  --gcc-install-dir DIR  Exact GCC directory, e.g. /usr/lib/gcc/aarch64-linux-gnu/10 (clang >= 16)

Options:
  --cxx CXX              C++ compiler (default: clang++)
  --business-cxx CXX     GCC 10 compiler used by the application side (default: g++)
  --business-flags FLAGS Extra flags for the application compiler, e.g. --sysroot=...
  --target TRIPLE        Cross-compile, e.g. aarch64-linux-gnu (default: native)
  --sysroot DIR          Sysroot for clang
  --arch ARCH            KFR_ARCH (default: neon64)
  --flags FLAGS          Extra compile flags (default: -ffp-contract=fast)
  --expect-gcc N         Required GCC major version of the selected toolchain (default: 10)
  --max-glibcxx VER      Newest allowed GLIBCXX symbol version (default: 3.4.28, GCC 10.3)
  --max-cxxabi VER       Newest allowed CXXABI symbol version (default: 1.3.12, GCC 10.3)
  --build-dir DIR        Build directory (default: ./build-kfr-clang)
  --prefix DIR           KFR install prefix (default: <build-dir>/install)
  --runner CMD           Command used to run target binaries, e.g. "qemu-aarch64 -L /usr/aarch64-linux-gnu"
                         (default: run natively unless --target is given)
  --no-run               Do not run the test binaries
  --jobs N               Parallel build jobs (default: nproc)
EOF
    exit 1
}

KFR_SRC=
GCC_TOOLCHAIN=
GCC_INSTALL_DIR=
CXX=clang++
BUSINESS_CXX=g++
BUSINESS_FLAGS=
TARGET=
SYSROOT=
ARCH=neon64
FLAGS="-ffp-contract=fast"
EXPECT_GCC=10
MAX_GLIBCXX=3.4.28
MAX_CXXABI=1.3.12
BUILD_DIR=$PWD/build-kfr-clang
PREFIX=
RUNNER=
RUN=1
JOBS=$(nproc)

while [ $# -gt 0 ]; do
    case "$1" in
    --kfr-src) KFR_SRC=$2; shift ;;
    --gcc-toolchain) GCC_TOOLCHAIN=$2; shift ;;
    --gcc-install-dir) GCC_INSTALL_DIR=$2; shift ;;
    --cxx) CXX=$2; shift ;;
    --business-cxx) BUSINESS_CXX=$2; shift ;;
    --business-flags) BUSINESS_FLAGS=$2; shift ;;
    --target) TARGET=$2; shift ;;
    --sysroot) SYSROOT=$2; shift ;;
    --arch) ARCH=$2; shift ;;
    --flags) FLAGS=$2; shift ;;
    --expect-gcc) EXPECT_GCC=$2; shift ;;
    --max-glibcxx) MAX_GLIBCXX=$2; shift ;;
    --max-cxxabi) MAX_CXXABI=$2; shift ;;
    --build-dir) BUILD_DIR=$2; shift ;;
    --prefix) PREFIX=$2; shift ;;
    --runner) RUNNER=$2; shift ;;
    --no-run) RUN=0 ;;
    --jobs) JOBS=$2; shift ;;
    -h | --help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

[ -n "$KFR_SRC" ] || usage
[ -n "$GCC_TOOLCHAIN$GCC_INSTALL_DIR" ] || usage
KFR_SRC=$(cd "$KFR_SRC" && pwd)
mkdir -p "$BUILD_DIR"
BUILD_DIR=$(cd "$BUILD_DIR" && pwd)
PREFIX=${PREFIX:-$BUILD_DIR/install}

step() { echo; echo "==== $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# Flags that make clang use the GCC 10 toolchain; needed for both compiling and linking
TC=()
[ -n "$TARGET" ] && TC+=("--target=$TARGET")
[ -n "$SYSROOT" ] && TC+=("--sysroot=$SYSROOT")
if [ -n "$GCC_INSTALL_DIR" ]; then
    TC+=("--gcc-install-dir=$GCC_INSTALL_DIR")
else
    TC+=("--gcc-toolchain=$GCC_TOOLCHAIN")
fi

if [ -n "$TARGET" ] && command -v "$TARGET-objdump" > /dev/null; then
    OBJDUMP=$TARGET-objdump
elif command -v llvm-objdump > /dev/null; then
    OBJDUMP=llvm-objdump
else
    OBJDUMP=objdump
fi

# ---------------------------------------------------------------------------------------------------------
step "1. Toolchain"
"$CXX" --version | head -1
SELECTED=$("$CXX" "${TC[@]}" -x c++ -E -v /dev/null 2>&1 | sed -n 's/^Selected GCC installation: //p')
[ -n "$SELECTED" ] || die "clang did not report a GCC installation for: ${TC[*]}"
CXX_INCLUDE=$("$CXX" "${TC[@]}" -x c++ -E -v /dev/null 2>&1 | grep -E '/include/c\+\+/[0-9.]+$' | head -1 | xargs)
SELECTED_VER=$(basename "$SELECTED")
echo "clang flags:          ${TC[*]}"
echo "selected GCC:         $SELECTED"
echo "libstdc++ headers:    $CXX_INCLUDE"
[ "${SELECTED_VER%%.*}" = "$EXPECT_GCC" ] ||
    die "clang selected GCC $SELECTED_VER, expected GCC $EXPECT_GCC. Use --gcc-install-dir, or a --gcc-toolchain prefix that contains only GCC $EXPECT_GCC"

BUSINESS_VER=$($BUSINESS_CXX $BUSINESS_FLAGS -dumpfullversion 2>/dev/null || $BUSINESS_CXX -dumpversion)
echo "application compiler: $BUSINESS_CXX ($BUSINESS_VER)"
[ "${BUSINESS_VER%%.*}" = "$EXPECT_GCC" ] || echo "WARNING: application compiler is not GCC $EXPECT_GCC"
BUSINESS_LIBSTDCXX=$(readlink -f "$($BUSINESS_CXX $BUSINESS_FLAGS -print-file-name=libstdc++.so)")
echo "application libstdc++: $BUSINESS_LIBSTDCXX (GCC 10.3 ships libstdc++.so.6.0.28)"

# ---------------------------------------------------------------------------------------------------------
step "2. KFR source"
KFR_VER=$(sed -n 's/^#define KFR_VERSION_\(MAJOR\|MINOR\|PATCH\) \([0-9]*\).*/\2/p' "$KFR_SRC/include/kfr/kfr.h" | paste -sd.)
echo "KFR version: ${KFR_VER:-unknown}"
[ "$KFR_VER" = "6.3.1" ] || echo "WARNING: this script was tested with KFR 6.3.1"
# patch paths are relative to the KFR root, which may be a subdirectory of another git repository
KFR_PREFIX=$(git -C "$KFR_SRC" rev-parse --show-prefix 2> /dev/null || true)
kfr_apply() { git -C "$KFR_SRC" apply ${KFR_PREFIX:+--directory="$KFR_PREFIX"} "$@"; }
PATCH=$SCRIPT_DIR/kfr-6.3.1-clang.patch
if kfr_apply --reverse --check "$PATCH" 2> /dev/null; then
    echo "clang patch already applied"
elif kfr_apply --check "$PATCH" 2> /dev/null; then
    kfr_apply "$PATCH"
    echo "applied $(basename "$PATCH")"
elif kfr_apply --reverse --check "$SCRIPT_DIR/../kfr-6.3.1-gcc-arm/kfr-6.3.1-gcc-arm.patch" 2> /dev/null; then
    echo "GCC/ARM patch is applied (it includes the clang fix)"
else
    die "cannot apply $PATCH to $KFR_SRC"
fi

# ---------------------------------------------------------------------------------------------------------
step "3. Build and install KFR"
GEN=()
command -v ninja > /dev/null && GEN=(-G Ninja)
CMAKE_ARGS=(
    -S "$KFR_SRC" -B "$BUILD_DIR/kfr" "${GEN[@]}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CXX_FLAGS="${TC[*]} $FLAGS"
    -DCMAKE_EXE_LINKER_FLAGS="${TC[*]}" -DCMAKE_SHARED_LINKER_FLAGS="${TC[*]}"
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DKFR_ARCH="$ARCH" -DKFR_ENABLE_DFT=ON -DENABLE_TESTS=OFF
)
if [ -n "$TARGET" ]; then
    CMAKE_ARGS+=(-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR="${TARGET%%-*}")
fi
cmake "${CMAKE_ARGS[@]}" > "$BUILD_DIR/kfr-configure.log" || die "cmake configure failed, see $BUILD_DIR/kfr-configure.log"
start=$(date +%s)
cmake --build "$BUILD_DIR/kfr" -j "$JOBS" > "$BUILD_DIR/kfr-build.log" 2>&1 || die "build failed, see $BUILD_DIR/kfr-build.log"
echo "built in $(($(date +%s) - start)) s"
cmake --install "$BUILD_DIR/kfr" > "$BUILD_DIR/kfr-install.log"
KFR_LIBS=$(find "$PREFIX" -name 'libkfr_dft*.a' -o -name 'libkfr_dsp*.a' | sort | tr '\n' ' ')
[ -n "$KFR_LIBS" ] || die "KFR libraries not found under $PREFIX"
echo "installed to $PREFIX: $(for l in $KFR_LIBS; do basename "$l"; done | tr '\n' ' ')"

# ---------------------------------------------------------------------------------------------------------
step "4. Library (clang) and application (GCC $EXPECT_GCC)"
D=$BUILD_DIR/abi_demo
mkdir -p "$D"
CLANG_CXX=("$CXX" "${TC[@]}" -std=c++17 -O3 -DNDEBUG $FLAGS -fPIC -I"$PREFIX/include" -I"$SCRIPT_DIR/abi_demo")
APP_CXX=($BUSINESS_CXX $BUSINESS_FLAGS -std=c++17 -O2 -I"$SCRIPT_DIR/abi_demo")

"${CLANG_CXX[@]}" -c "$SCRIPT_DIR/abi_demo/fft_wrapper.cpp" -o "$D/fft_wrapper.o"
rm -f "$D/libfft_wrapper.a"
ar rcs "$D/libfft_wrapper.a" "$D/fft_wrapper.o"
# shared library linked by clang against the GCC 10 runtime, KFR linked in statically
"$CXX" "${TC[@]}" -shared -o "$D/libfft_wrapper.so" "$D/fft_wrapper.o" $KFR_LIBS -Wl,--no-undefined -lm
"${CLANG_CXX[@]}" -DKFR_HAVE_DFT -Wno-deprecated-declarations -c "$SCRIPT_DIR/../kfr-6.3.1-gcc-arm/check.cpp" -o "$D/check.o"

"${APP_CXX[@]}" -c "$SCRIPT_DIR/abi_demo/app.cpp" -o "$D/app.o"
"${APP_CXX[@]}" "$D/app.o" "$D/libfft_wrapper.a" $KFR_LIBS -o "$D/app_static" -lm
"${APP_CXX[@]}" "$D/app.o" -L"$D" -lfft_wrapper -Wl,-rpath,"$D" -o "$D/app_shared" -lm
"${APP_CXX[@]}" "$D/check.o" $KFR_LIBS -o "$D/check" -lm
echo "app_static: application (GCC) + libfft_wrapper.a (clang) + KFR (clang), linked by GCC"
echo "app_shared: application (GCC) + libfft_wrapper.so (clang, KFR inside)"
echo "check:      KFR correctness test compiled by clang, linked by GCC"

# ---------------------------------------------------------------------------------------------------------
step "5. ABI check (GLIBCXX <= $MAX_GLIBCXX, CXXABI <= $MAX_CXXABI)"
version_le() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$2" ]; }
ABI_FAIL=0
for f in "$D/libfft_wrapper.so" "$D/app_static" "$D/app_shared" "$D/check"; do
    syms=$($OBJDUMP -T "$f")
    g=$(grep -oE 'GLIBCXX_[0-9.]+' <<< "$syms" | sed 's/GLIBCXX_//' | sort -Vu | tail -1)
    c=$(grep -oE 'CXXABI_[0-9.]+' <<< "$syms" | sed 's/CXXABI_//' | sort -Vu | tail -1)
    status=OK
    if [ -n "$g" ] && ! version_le "$g" "$MAX_GLIBCXX"; then status=FAIL; fi
    if [ -n "$c" ] && ! version_le "$c" "$MAX_CXXABI"; then status=FAIL; fi
    printf '%-18s newest GLIBCXX=%-8s CXXABI=%-8s %s\n' "$(basename "$f")" "${g:--}" "${c:--}" "$status"
    if [ $status = FAIL ]; then
        ABI_FAIL=1
        grep -E 'GLIBCXX_|CXXABI_' <<< "$syms" | while read -r line; do
            v=$(grep -oE '(GLIBCXX|CXXABI)_[0-9.]+' <<< "$line")
            case $v in
            GLIBCXX_*) version_le "${v#GLIBCXX_}" "$MAX_GLIBCXX" || echo "    $line" ;;
            CXXABI_*) version_le "${v#CXXABI_}" "$MAX_CXXABI" || echo "    $line" ;;
            esac
        done
    fi
done
[ $ABI_FAIL = 0 ] || die "binaries require libstdc++ symbols newer than GCC $EXPECT_GCC provides"

# ---------------------------------------------------------------------------------------------------------
if [ $RUN = 1 ]; then
    step "6. Run"
    if [ -z "$RUNNER" ] && [ -n "$TARGET" ]; then
        echo "cross build: pass --runner to run the binaries, skipping"
    else
        for b in app_static app_shared check; do
            echo "-- $b"
            $RUNNER "$D/$b" | grep -E 'library|app:|KFR|FAIL|PASSED'
        done
    fi
fi

step "Done"
echo "KFR:            $PREFIX"
echo "example output: $D"
