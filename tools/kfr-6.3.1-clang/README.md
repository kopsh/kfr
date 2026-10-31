# KFR 6.3.1：用 Clang + GCC 10 工具链构建

适用场景：FFT 库用 Clang 编译（KFR 的主力编译器，性能和编译时间都最好），交给用 **GCC 10.3.1** 编译的业务软件链接。

做法：Clang 通过 `--gcc-toolchain` 或 `--gcc-install-dir` 使用 GCC 10 的 libstdc++ 头文件和运行时库，编出来的库和 GCC 10 业务代码用的是同一套标准库实现。`build-clang.sh` 会完成整个构建，并检查 ABI 是否兼容。

## 前提

业务代码**不直接包含 KFR 的头文件**。KFR 只在你们的库内部使用，对外只暴露你们自己的接口，例如 `abi_demo/fft_wrapper.h` 这样的 C 接口。

- 接口里也可以用 `std::vector`、`std::complex` 等标准库类型，两边是同一个 libstdc++。但两边的 `_GLIBCXX_USE_CXX11_ABI` 必须相同，默认都是 1。
- 不要让异常跨越接口抛出。

## 用法

```bash
# 在 ARM64 机器上本地构建，GCC 10.3.1 单独安装在 /opt/gcc-10.3.1
./build-clang.sh --kfr-src /path/to/kfr-6.3.1 \
                 --gcc-toolchain /opt/gcc-10.3.1 \
                 --business-cxx /opt/gcc-10.3.1/bin/g++

# 同一前缀下装了多个 GCC 版本时，用 --gcc-install-dir 精确指定（需要 clang >= 16）
./build-clang.sh --kfr-src /path/to/kfr-6.3.1 \
                 --gcc-install-dir /usr/lib/gcc/aarch64-linux-gnu/10 \
                 --business-cxx g++-10

# 交叉编译，用 qemu 运行
./build-clang.sh --kfr-src /path/to/kfr-6.3.1 --target aarch64-linux-gnu \
                 --gcc-install-dir /usr/lib/gcc-cross/aarch64-linux-gnu/10 \
                 --business-cxx aarch64-linux-gnu-g++-10 \
                 --runner "qemu-aarch64 -L /usr/aarch64-linux-gnu"
```

其他选项见 `./build-clang.sh --help`。

## 脚本做了什么

1. **确认工具链**：读取 Clang 实际选中的 GCC 安装目录（`Selected GCC installation`），主版本号不是 10 就报错退出。
   - 注意：Clang 会自动选择前缀下**版本最高**的 GCC。如果 `--gcc-toolchain` 指向的前缀里同时装了 GCC 10 和更新的版本，选中的会是新版本。
2. **打补丁**：自动应用 `kfr-6.3.1-clang.patch`，内容是 `select.hpp` 中 NEON 掩码的类型转换，新版 Clang 不加这个会报错。重复执行不会重复打。KFR 源码放在其他 git 仓库的子目录里也可以。
3. **编译 KFR**：用 Clang 编译并安装 KFR（Release，开启 DFT，PIC）。
4. **构建示例**：
   - `libfft_wrapper.a` / `libfft_wrapper.so`：Clang 编译的示例库，提供 C 接口，KFR 静态链接在里面。
   - `app_static` / `app_shared`：GCC 10 编译和链接的业务程序，分别静态链接、动态链接上面的库。
   - `check`：KFR 的正确性测试，用 Clang 编译、GCC 10 链接。
5. **ABI 检查**：检查上述二进制文件依赖的 libstdc++ 符号版本，不能超过 GCC 10.3 提供的上限（`GLIBCXX_3.4.28`、`CXXABI_1.3.12`）。超出时列出具体符号，并以失败退出。
6. **运行**：运行三个程序，和直接实现的 DFT 结果对比。

## 验证结果

测试环境：clang 18.1，GCC 10.5（aarch64 交叉编译），在 qemu 上运行。

```
==== 5. ABI check (GLIBCXX <= 3.4.28, CXXABI <= 1.3.12)
libfft_wrapper.so  newest GLIBCXX=3.4.21   CXXABI=1.3      OK
app_static         newest GLIBCXX=3.4.21   CXXABI=1.3.9    OK
app_shared         newest GLIBCXX=3.4      CXXABI=1.3.9    OK
check              newest GLIBCXX=3.4.21   CXXABI=1.3      OK
```

三个程序都输出 `PASSED: 0 failures`。

另外验证了两个反例：
- `--gcc-toolchain=/usr` 下同时装有 GCC 10 和 13 时，Clang 选中了 13，脚本报错退出。
- 用 GCC 13 编译、调用了 `std::to_chars(double)` 的程序依赖 `GLIBCXX_3.4.29`，被正确判定为超出上限。

## 接入你们自己的库

- 编译选项：使用和脚本相同的 Clang 选项，即 `--gcc-toolchain` 或 `--gcc-install-dir`、`-std=c++17 -O3 -DNDEBUG`。链接时也要带上 `--gcc-*` 选项。
- 交付前检查：把脚本第 5 步的检查加到 CI 里，下面这条命令应该没有任何输出：

  ```bash
  objdump -T 你的库或程序 | grep -E 'GLIBCXX_3\.4\.(29|[3-9][0-9])|CXXABI_1\.3\.(1[3-9])'
  ```

- 静态库（`.a`）没有符号版本信息。要用 GCC 10 链接一个测试程序，再检查生成的可执行文件，脚本里的 `app_static` 就是这么做的。
