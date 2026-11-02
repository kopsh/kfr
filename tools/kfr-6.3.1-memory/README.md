# KFR 6.3.1：大点数 DFT 的内存占用分析

问题：200M 点的 DFT 占用约 10 GB 内存。

## 内存都用在哪里

`dft_plan<T>(N)` 除了用户自己的输入、输出缓冲区以外，还需要两块大小约为 N 个复数的内存：

| 内存块 | 大小 | 来源 |
|---|---|---|
| `plan.data`（旋转因子表） | ≈ N × sizeof(complex&lt;T&gt;) | 每个 stage 存 `repeats × (radix−1)` 个旋转因子，各 stage 加起来正好约等于 N−1 |
| `plan.temp_size`（scratch） | ≈ N × sizeof(complex&lt;T&gt;) | 各 stage 都是 out-of-place 计算，最后还有一个 reorder stage，所以中间结果需要一块乒乓缓冲区 |

以 200,000,000 = 2⁹·5⁸ 为例，KFR 把它分解为 1 个 radix-2 stage 和 8 个 radix-10 stage，再加 1 个 reorder stage。其中第一个 stage 的旋转因子表就有 N/2 个复数（f32 下 800 MB），第二个 stage 有 0.45N 个（720 MB），其余 stage 加起来不到 80 MB。

200M 点的实测结果（`probe.cpp`）：

| | plan.data | temp | 峰值 RSS |
|---|---|---|---|
| 复数 f32，out-of-place | 1.49 GiB | 1.49 GiB | 5.96 GiB |
| 复数 f32，in-place | 1.49 GiB | 1.49 GiB | 4.48 GiB |
| 复数 f64，in-place | 2.98 GiB | 2.98 GiB | 8.95 GiB |
| 复数 f64，out-of-place | 2.98 GiB | 2.98 GiB | 11.9 GiB（按公式推算） |
| 实数 f64（`dft_plan_real`） | 2.24 GiB | 1.49 GiB | 6.71 GiB |

峰值 RSS = 用户缓冲区 + plan.data + temp。plan.data 和 temp 的大小在 aarch64（Clang，neon64，qemu）和 x86 上完全一致。

另外两点需要注意：
- 用 `kfr::dft()` / `idft()` 这类便捷函数时，plan 会被 `dft_cache` 一直缓存，函数返回后 plan.data 也不会释放。
- 如果 N 分解掉 2–10 的因子后，剩下的部分 ≥ 101（即含有较大的质因子），KFR 会改用 Bluestein 算法，内部再建一个 next_pow2(2N−1) 点的 plan，内存会再翻好几倍。

## 不改代码就能省的

1. **in-place 执行**：`out == in`，可以省掉一个缓冲区。
2. **实数输入用 `dft_plan_real`**：内存约减半。
3. **不要用 `dft()` 便捷函数**：自己创建 plan，用完就释放。多个 plan 可以共用同一块 temp。

## 四步法（four-step）：额外内存从 2N 降到几 MB，速度更快

`fourstep.hpp` 只用 KFR 的公开接口实现，不需要改 KFR 源码：

- 把 N 分解为 N1 × N2。先对 N2 列分别做 N1 点 DFT，再乘旋转因子 ω_N^(n2·k1)，最后对 N1 行分别做 N2 点 DFT，并转置写到输出。
- 旋转因子由两张约 √N 大小的表现算。KFR 只需要创建 N1 点和 N2 点两个小 plan。
- **N1×N2 的选择**：建 plan 时在当前机器上测量各候选长度子 FFT 的单点耗时，取总耗时最小的组合（约 0.2 s）。不同长度的子 FFT 单点耗时能差 2 倍以上，而且排名和 CPU 有关，所以要在目标机器上现场测。也可以用 `options.n1` 固定。
- 列读取和转置写回时按行做软件预取。
- 三种调用方式：

| 接口 | 说明 | 峰值内存 |
|---|---|---|
| `execute_destroy(out, in)` | out-of-place，`in` 被当作工作区覆盖 | in + out |
| `execute_inplace(x)` | 原地；内部分配一块 N 点工作区（也可以传入） | x + 工作区 |
| `execute(out, in)` | out-of-place，保留 `in`；需要一块 N 点工作区 | in + out + 工作区 |

输出顺序和 `dft_plan::execute` 相同（自然顺序），逆变换同样不做 1/N 归一化。

### 200M 点实测（x86 单线程，每项 5 次取中位数）

- "native" 表示 KFR 运行时选 AVX-512。
- "sse41" 表示强制使用 128 位 SIMD，和 NEON 宽度相同，作为 ARM 的近似。
- KFR 直接计算的时间不含建 plan（f32 约 7–14 s，f64 约 7–23 s）。四步法建 plan 约 0.2 s，含测量。

| 类型 | 模式 | KFR 直接计算 | 四步法 | 内存 |
|---|---|---|---|---|
| f32 native | out-of-place | 5.32 s / 5.97 GiB | 3.62 s / 3.01 GiB（覆盖输入） | −50% |
| f32 native | 保留输入 | 5.32 s / 5.97 GiB | 3.52 s / 4.49 GiB | −25% |
| f32 native | 原地 | 5.13 s / 4.47 GiB | 4.15 s / 3.01 GiB | −33% |
| f64 native | out-of-place | 8.43 s / 11.93 GiB | 5.09 s / 5.98 GiB（覆盖输入） | −50% |
| f64 native | 保留输入 | 8.43 s / 11.93 GiB | 5.33 s / 8.96 GiB | −25% |
| f64 native | 原地 | 8.00 s / 8.95 GiB | 5.29 s / 5.98 GiB | −33% |
| f32 sse41 | out-of-place | 6.30 s / 5.97 GiB | 4.33 s / 3.00 GiB（覆盖输入） | −50% |
| f32 sse41 | 保留输入 | 6.30 s / 5.97 GiB | 4.38 s / 4.49 GiB | −25% |
| f32 sse41 | 原地 | 6.16 s / 4.47 GiB | 4.37 s / 3.00 GiB | −33% |
| f64 sse41 | out-of-place | 9.55 s / 11.93 GiB | 6.47 s / 5.98 GiB（覆盖输入） | −50% |
| f64 sse41 | 保留输入 | 9.55 s / 11.93 GiB | 6.71 s / 8.97 GiB | −25% |
| f64 sse41 | 原地 | 9.30 s / 8.95 GiB | 6.50 s / 5.98 GiB | −33% |

四步法快的原因：KFR 直接计算时，200M 点要经过 10 个 stage，每个 stage 都把全部数据从内存读写一遍，前两个 stage 还要额外读约 0.95N 个旋转因子。四步法只读写两遍，子 FFT 都在缓存里完成。这台机器的 L3 有 260 MB，掩盖了一部分访存开销；ARM 服务器的 L3 更小、单核带宽更低，预计优势不会比这里小，但必须在真机上确认。

精度（f32/f64 结果对比 f64 直接计算的相对 RMS 误差）：四步法和 KFR 直接计算在同一量级。

| N | 类型 | KFR 直接计算 | 四步法（三种模式相同） | 正反变换往返 |
|---|---|---|---|---|
| 1M | f32 | 2.41e-7 | 2.43e-7 | 2.79e-7 |
| 2M | f32 | 2.33e-7 | 2.30e-7 | 2.63e-7 |
| 2M | f64 | 参考 | 4.62e-16 | 4.63e-16 |
| 50M | f32 | 2.73e-7 | 2.76e-7 | 3.08e-7 |

aarch64（Clang + neon64，qemu）：2M（f32/f64）和 16M 点，三种模式全部通过；200M 点峰值 RSS 和 x86 一致。

### 什么时候用

- **N 不是 2 的幂**（例如 200M）：用四步法。
- **N 是 2 的幂**：继续用 KFR，并且原地计算。KFR 对 2 的幂走专门的代码路径，原地计算不需要 temp。2²⁷ 点 f32 实测：KFR 原地 1.72 s / 2.00 GiB，四步法原地 2.21 s / 2.01 GiB，四步法反而慢。
- **找不到合适的拆分**：`fft_fourstep<T>::find_split(n)` 返回 0（例如 N 含大于 100 的质因子），这时退回 `dft_plan`。

### 限制

- 列读取、旋转因子乘法和转置还是标量代码，依赖编译器自动向量化，没有手写 NEON。
- 还没有做多线程。各列、各行之间互相独立，可以直接并行。
- 性能必须在 ARM 真机上用 `run_bench.sh` 确认。

## KFR 最新版（7.x）

- 新增了一个基于四步法思路的 FFT 实现（`src/dft/sandwich.hpp`，`ngfft`），但只支持 **2 的幂**，而且需要 **C++20**。它中间那一步的旋转因子仍然按 N 个存储（`sandwich_twiddle_size` 里的 `r1 * r2`）。
- **非 2 的幂的路径**（`init_dft`）和 6.3.1 基本相同，只是因子分解改成了质因数分解。每个 stage 仍有自己的旋转因子表，总量约 N，scratch 也仍是 N。
- 结论：升级到 7.x 解决不了 200M 点的内存问题，而且你们的 C++17 环境也用不了 7.x。

## 文件

| 文件 | 说明 |
|---|---|
| `probe.cpp` | 打印 plan.data、temp_size 和各 stage（`DUMP=1`），测峰值 RSS |
| `fourstep.hpp` | 四步法实现：`execute_destroy`（覆盖输入）、`execute`（保留输入）、`execute_inplace`（原地） |
| `bench.cpp` | 精度检查（`check`），以及 KFR 直接计算和四步法各模式的内存、耗时对比 |
| `run_bench.sh` | 在目标机器上编译 `bench.cpp` 并跑完整对比 |

```bash
# aarch64，KFR 由 ../kfr-6.3.1-clang/build-clang.sh 安装到 $KFR
./run_bench.sh --kfr $KFR --cxx clang++ --n 200000000
```
