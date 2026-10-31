#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <cmath>
#include <complex>
#include <cstdio>

using namespace kfr;

static int failures = 0;

template <typename T>
static void check_dft(size_t n)
{
    univector<complex<T>> in(n), out(n), back(n);
    for (size_t i = 0; i < n; i++)
        in[i] = complex<T>(std::sin(T(i) * T(0.37)) + T(0.1) * T(i % 7), std::cos(T(i) * T(0.11)));
    dft_plan<T> plan(n);
    univector<u8> tmp(plan.temp_size);
    plan.execute(out, in, tmp);
    plan.execute(back, out, tmp, true);
    double maxerr = 0, maxerr_inv = 0;
    for (size_t k = 0; k < n; k++)
    {
        std::complex<double> s = 0;
        for (size_t i = 0; i < n; i++)
            s += std::complex<double>(in[i].real(), in[i].imag()) *
                 std::polar(1.0, -2.0 * M_PI * double(i * k % n) / double(n));
        maxerr     = std::max(maxerr, std::abs(s - std::complex<double>(out[k].real(), out[k].imag())));
        maxerr_inv = std::max(maxerr_inv, std::abs(std::complex<double>(back[k].real(), back[k].imag()) / double(n) -
                                                   std::complex<double>(in[k].real(), in[k].imag())));
    }
    double tol = (sizeof(T) == 4 ? 2e-4 : 1e-10) * n;
    bool ok    = maxerr < tol && maxerr_inv < tol;
    if (!ok)
        failures++;
    printf("dft<%s> n=%-5zu err=%.3g inv_err=%.3g %s\n", sizeof(T) == 4 ? "f32" : "f64", n, maxerr, maxerr_inv,
           ok ? "OK" : "FAIL");
}

template <typename T>
static void check_real_dft(size_t n)
{
    univector<T> in(n), back(n);
    univector<complex<T>> out(n / 2 + 1);
    for (size_t i = 0; i < n; i++)
        in[i] = std::sin(T(i) * T(0.21)) + T(i % 5);
    dft_plan_real<T> plan(n);
    univector<u8> tmp(plan.temp_size);
    plan.execute(out, in, tmp);
    plan.execute(back, out, tmp);
    double err = 0;
    for (size_t k = 0; k <= n / 2; k++)
    {
        std::complex<double> s = 0;
        for (size_t i = 0; i < n; i++)
            s += double(in[i]) * std::polar(1.0, -2.0 * M_PI * double(i * k % n) / double(n));
        err = std::max(err, std::abs(s - std::complex<double>(out[k].real(), out[k].imag())));
    }
    double inv = 0;
    for (size_t i = 0; i < n; i++)
        inv = std::max(inv, std::abs(double(back[i]) / n - double(in[i])));
    double tol = (sizeof(T) == 4 ? 2e-4 : 1e-10) * n;
    bool ok    = err < tol && inv < tol;
    if (!ok)
        failures++;
    printf("real_dft<%s> n=%-5zu err=%.3g inv_err=%.3g %s\n", sizeof(T) == 4 ? "f32" : "f64", n, err, inv,
           ok ? "OK" : "FAIL");
}

template <typename T>
static void check_dsp()
{
    // FIR: compare against direct convolution
    univector<T> taps(31), x(500), y(500);
    fir_lowpass(taps, 0.2, to_handle(window_kaiser<T>(taps.size(), 3.0)), true);
    for (size_t i = 0; i < x.size(); i++)
        x[i] = std::sin(T(i) * T(0.3)) + T(i % 3);
    y = fir(x, fir_params<T>(taps));
    double err = 0;
    for (size_t i = 0; i < x.size(); i++)
    {
        double s = 0;
        for (size_t j = 0; j < taps.size() && j <= i; j++)
            s += double(taps[j]) * x[i - j];
        err = std::max(err, std::abs(s - y[i]));
    }
    bool ok = err < (sizeof(T) == 4 ? 1e-4 : 1e-12);
    failures += !ok;
    printf("fir<%s> err=%.3g %s\n", sizeof(T) == 4 ? "f32" : "f64", err, ok ? "OK" : "FAIL");

    // Biquad: compare against direct form I
    biquad_section<T> bq[] = { biquad_lowpass(0.1, 0.7), biquad_highpass(0.02, 0.5) };
    univector<T> z(x.size());
    z = biquad(bq, x);
    double s1[2][4] = {};
    double berr     = 0;
    for (size_t i = 0; i < x.size(); i++)
    {
        double v = x[i];
        for (int k = 0; k < 2; k++)
        {
            double* s  = s1[k];
            double out = bq[k].b0 * v + bq[k].b1 * s[0] + bq[k].b2 * s[1] - bq[k].a1 * s[2] - bq[k].a2 * s[3];
            s[1] = s[0], s[0] = v, s[3] = s[2], s[2] = out;
            v = out;
        }
        berr = std::max(berr, std::abs(v - z[i]));
    }
    ok = berr < (sizeof(T) == 4 ? 1e-3 : 1e-10);
    failures += !ok;
    printf("biquad<%s> err=%.3g %s\n", sizeof(T) == 4 ? "f32" : "f64", berr, ok ? "OK" : "FAIL");

    // Elementwise math
    univector<T> a = truncate(linspace(T(0.01), T(10), 257), 257);
    univector<T> r = sin(a) * cos(a) + exp(-a) + log(a) + sqrt(a);
    double merr = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        double v = a[i];
        merr     = std::max(merr, std::abs(std::sin(v) * std::cos(v) + std::exp(-v) + std::log(v) + std::sqrt(v) - r[i]));
    }
    ok = merr < (sizeof(T) == 4 ? 1e-4 : 1e-9);
    failures += !ok;
    printf("math<%s> err=%.3g %s\n", sizeof(T) == 4 ? "f32" : "f64", merr, ok ? "OK" : "FAIL");
}

int main()
{
    printf("%s\n", library_version());
    for (size_t n : { 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 3, 5, 7, 12, 30, 100, 210, 1000, 997 })
    {
        check_dft<float>(n);
        check_dft<double>(n);
    }
    for (size_t n : { 4, 16, 64, 256, 1024, 30, 100, 1000 })
    {
        check_real_dft<float>(n);
        check_real_dft<double>(n);
    }
    check_dsp<float>();
    check_dsp<double>();
    printf("%s: %d failures\n", failures ? "FAILED" : "PASSED", failures);
    return failures != 0;
}
