# KFR 6.3.1：GCC / ARM64 支持

KFR 6.3.1（C++17）原本只支持用 Clang 编译 ARM 版本。本目录提供一个补丁，让 GCC 10 / 12 也能编译 ARM64（NEON）版本，DFT 模块包含在内。另外附带正确性测试程序和目标码统计脚本。

> 如果业务软件只是链接你们的库，而不直接包含 KFR 头文件，那么更推荐用 Clang 编译 KFR 和你们的库，再交给 GCC 10 的业务软件链接。见 [`../kfr-6.3.1-clang`](../kfr-6.3.1-clang/README.md)。本补丁作为必须用 GCC 时的备选方案。

## 文件

| 文件 | 说明 |
|---|---|
| `kfr-6.3.1-gcc-arm.patch` | 在原版 KFR 6.3.1 源码根目录执行 `git apply` |
| `check.cpp` | 正确性测试：复数/实数 DFT（2 的幂、质数、混合基长度）、FIR、biquad、初等函数，和 double 精度的直接实现对比 |
| `asmstats.sh` | 统计 aarch64 目标文件的指令分布（mov、`[sp]`、zip/uzp/trn/ext/rev64、tbl 等） |

## 用法

```bash
git clone --branch 6.3.1 https://github.com/kfrlib/kfr.git kfr-6.3.1
cd kfr-6.3.1 && git apply /path/to/kfr-6.3.1-gcc-arm.patch
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 \
      -DKFR_ARCH=neon64 -DENABLE_TESTS=OFF
ninja -C build
```

打上补丁后，GCC 在非 x86 平台上默认就开启 DFT，不需要再加 `-DKFR_ENABLE_DFT=ON`。

## 补丁内容

1. **启用向量扩展后端**：GCC 在非 x86 平台上走向量扩展后端（`CMT_CLANG_EXT`），不再走 `backend_generic`。后者在 NEON 下没有维护，混有 SSE 代码。
2. **宽向量的表示**：16 字节以内的向量用原生的 `vector_size` 类型。更宽的向量用 `simd_chunks`，即由高低两半组成、逐层嵌套的结构体。
   原因：AArch64 上没有比 128 位更宽的向量寄存器模式，GCC 遇到超宽的 `vector_size` 类型时，部分读写、拼接和 shuffle 都会被拆成逐元素操作并经过栈。
3. **shuffle**：在编译期拆成若干个 128 位的双输入置换，每个输出块只取 1–2 个源块。
   - 需要两条指令才能完成的 4×32 位置换（例如 `<0,5,2,7>`），GCC 会退化成 `tbl`。补丁在编译期搜索出由两条可识别指令（zip/uzp/trn/ext/rev64/dup/ins）组成的等价组合。
   - GCC 12 以下没有 `__builtin_shufflevector`，改用 `__builtin_shuffle`。另外，64 位成对置换先转换成 64 位 lane 再做，`ins` 模式直接写单个元素。
4. **非对齐读写**：`simd_chunks` 不是 POD，`packed` 属性对它无效，所以改为按原生向量的半块读写。
5. **NEON 细节**：`select` 的掩码加上 `vreinterpretq`；GCC 没有 `__builtin_bitreverse32`，改用 `rbit` 指令。
6. **CMake**（只对 GCC + 非 x86 生效）：
   - `-flax-vector-conversions`：作为 INTERFACE 选项传给使用 KFR 的代码。
   - KFR 自己的目标额外加两个选项：
     - `-fno-tree-loop-if-convert`：编译时间约减少 40%，生成的代码完全相同。
     - `-fno-schedule-insns`：寄存器分配之前的第一遍调度会推高寄存器压力，关掉后 spill 约减少 43%。

Clang 的代码路径不受影响。

## 验证结果

测试环境：交叉编译，`-O3 -DNDEBUG -ffp-contract=fast -mcpu=cortex-a72`，在 qemu-aarch64 上运行。

- KFR 官方单元测试（GCC 12.4）：200/200 个测试、2,824,716 项检查全部通过。
- `check.cpp`：GCC 10.5、GCC 12.4、Clang 18.1 全部通过。

`fft-impl-f32.o` 的目标码统计（`asmstats.sh`）：

| | 总指令 | mov | `[sp]` | tbl |
|---|---|---|---|---|
| GCC 12，向量直接用 `vector_size`（旧做法） | 135,232 | 15,301 | 32,837 | 854 |
| GCC 12，本补丁 | 53,358 | 3,090 | 5,074 | 0 |
| GCC 10，本补丁 | 54,331 | 8,589 | 4,347 | 5 |
| Clang 18，原版 | 50,028 | 4,196 | 2,864 | 0 |

`fft-impl-f64.o`：GCC 12 + 本补丁为 33,619 条指令、`[sp]` 2,173；Clang 为 36,818 条、`[sp]` 1,890。

编译时间（CMake 编库，13 个编译单元，4 核）：

| | 旧做法 | 本补丁 |
|---|---|---|
| GCC 12 | 64 s | 116 s |
| GCC 10 | 174 s | 453 s |

GCC 10 慢主要是因为 IPA 内联启发式和跨过程常量传播，所以建议优先用 GCC 12。

## 已知限制

- 以上都是静态统计，实际性能需要在 ARM 真机上测，例如用 fft-benchmark `a7d0306`。
- 没有测过 32 位 armv7，也没有测过 GCC 11、GCC 13 及以上版本。
