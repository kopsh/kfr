/*
 * "Business" application: compiled and linked with GCC 10, uses the Clang-built FFT library
 * through its C interface and libstdc++ on its own side.
 */
#include "fft_wrapper.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

static int check(size_t n)
{
    std::vector<std::complex<float>> in(n), out(n), back(n);
    for (size_t i = 0; i < n; ++i)
        in[i] = { std::sin(0.37f * i) + 0.1f * (i % 7), std::cos(0.11f * i) };

    fft_wrapper_plan* plan = fft_wrapper_create_f32(n);
    if (!plan)
        return 1;
    const float* pin = reinterpret_cast<const float*>(in.data());
    if (fft_wrapper_forward_f32(plan, reinterpret_cast<float*>(out.data()), pin) != 0 ||
        fft_wrapper_inverse_f32(plan, reinterpret_cast<float*>(back.data()),
                                reinterpret_cast<const float*>(out.data())) != 0)
        return 1;
    fft_wrapper_destroy(plan);

    double err = 0, inv_err = 0;
    for (size_t k = 0; k < n; ++k)
    {
        std::complex<double> s = 0;
        for (size_t i = 0; i < n; ++i)
            s += std::complex<double>(in[i]) * std::polar(1.0, -2.0 * M_PI * double(i * k % n) / double(n));
        err     = std::max(err, std::abs(s - std::complex<double>(out[k])));
        inv_err = std::max(inv_err, std::abs(std::complex<double>(back[k]) / double(n) - std::complex<double>(in[k])));
    }
    const bool ok = err < 2e-4 * n && inv_err < 2e-4 * n;
    std::printf("n=%-5zu err=%.3g inv_err=%.3g %s\n", n, err, inv_err, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main()
{
    std::printf("library: %s\n", fft_wrapper_version());
#if defined(__GNUC__) && !defined(__clang__)
    std::printf("app:     gcc-%d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#endif
    int failures = 0;
    for (size_t n : { 16, 256, 1024, 4096, 12, 100, 1000, 997 })
        failures += check(n);
    std::printf("%s: %d failures\n", failures ? "FAILED" : "PASSED", failures);
    return failures != 0;
}
