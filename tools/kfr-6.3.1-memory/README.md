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

## 四步法（four-step）：额外内存从 2N 降到几 MB

`fourstep.hpp` 只用 KFR 的公开接口实现，不需要改 KFR 源码：

- 把 N 分解为 N1 × N2（都接近 √N）。
- 先对 N2 列分别做 N1 点 DFT，再乘旋转因子 ω_N^(n2·k1)，最后对 N1 行分别做 N2 点 DFT，并转置写到输出。
- 旋转因子由两张约 √N 大小的表相乘得到。
- KFR 只需要创建 N1 点和 N2 点两个小 plan。

200M 点，x86 单线程（`fstest.cpp`）：

| | 额外内存 | 峰值 RSS | 建 plan | 执行 |
|---|---|---|---|---|
| KFR 直接计算，f32 | 2.98 GiB | 5.96 GiB | 20.1 s | 5.19 s |
| 四步法，f32 | 4.4 MiB | 2.99 GiB | 0.01 s | 4.77 s |
| 四步法，f64 | 8.7 MiB | 5.97 GiB | 0.05 s | 6.23 s |

精度（f32 结果和 f64 直接计算对比的相对 RMS 误差）：

| N | KFR 直接计算 | 四步法 | 四步法正反变换往返 |
|---|---|---|---|
| 1M | 2.41e-7 | 2.42e-7 | 2.80e-7 |
| 4M（2²²） | 1.45e-7 | 1.61e-7 | 1.99e-7 |
| 50M | 2.73e-7 | 2.76e-7 | 3.08e-7 |

aarch64（Clang + neon64，qemu）：2M 点精度和 x86 一致；200M 点四步法的峰值 RSS 为 3.0 GiB。qemu 下的耗时没有参考价值。

### 限制和后续工作

- 当前接口是 out-of-place，**输入缓冲区会被当作工作区覆盖**。如果必须保留输入，需要再分配一块 N 点缓冲区，但仍然比直接计算少一块。
- 原型里的列读取、旋转因子乘法和转置都是标量代码。上线前可以做的优化：
  - 用 NEON 向量化，调整块大小 `B`；
  - 各列之间、各行之间互相独立，可以直接多线程并行。
- 输出顺序和 `dft_plan::execute` 相同（自然顺序），逆变换同样不做 1/N 归一化。
- 性能需要在 ARM 真机上测。

## 文件

| 文件 | 说明 |
|---|---|
| `probe.cpp` | 打印 plan.data、temp_size 和各 stage（`DUMP=1`），测峰值 RSS |
| `fourstep.hpp` | 四步法实现 |
| `fstest.cpp` | 精度测试（`accuracy`），以及直接计算和四步法的内存、耗时对比（`direct` / `fourstep`，可选 `f64`） |

```bash
# aarch64，KFR 由 ../kfr-6.3.1-clang/build-clang.sh 安装到 $KFR
clang++ -std=c++17 -O3 -DNDEBUG -I$KFR/include fstest.cpp -L$KFR/lib -lkfr_dft_neon64 -o fstest
./fstest 2000000 accuracy
./fstest 200000000 direct
./fstest 200000000 fourstep
```
