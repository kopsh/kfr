// Memory / speed / accuracy comparison: KFR dft_plan<T>(N) vs fft_fourstep<T>.
//
//   bench N f32|f64 MODE [reps]          time and peak RSS (input regenerated before every run, not timed)
//   bench N f32|f64 check                accuracy of all modes vs f64 dft_plan, plus inverse
// MODE: direct | direct_inplace | fs (out-of-place, input preserved) | fs_inplace (no work buffer)
//       | fs_inplace_work (in-place using an N-point work buffer)
// env:  CPU=sse2|sse41|avx|avx2 (x86 only, override KFR's runtime dispatch), FS_N1, FS_B, FS_PF (options)
#include "fourstep.hpp"
#ifdef __x86_64__
#include <kfr/runtime/cpuid_auto.hpp>
#endif
#include <sys/resource.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace kfr;

static double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double peak_gib()
{
    rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1048576.0;
}
template <typename T>
static void fill(complex<T>* a, size_t n)
{
    for (size_t i = 0; i < n; i++)
        a[i] = complex<T>(T(std::sin(0.001 * i) + ((i * 2654435761u) >> 28) * 0.01), T(std::cos(0.37 * i)));
}
template <typename T>
static typename fft_fourstep<T>::options fs_opts()
{
    typename fft_fourstep<T>::options o;
    if (const char* e = getenv("FS_N1"))
        o.n1 = strtoull(e, nullptr, 10);
    if (const char* e = getenv("FS_B"))
        o.block = strtoull(e, nullptr, 10);
    if (const char* e = getenv("FS_PF"))
        o.prefetch = strtoull(e, nullptr, 10);
    return o;
}

template <typename T>
static int timing(size_t n, const std::string& mode, int reps)
{
    univector<complex<T>> in(n), out(mode.find("inplace") != std::string::npos ? 0 : n);
    if (out.size())
        std::memset((void*)out.data(), 0, n * sizeof(complex<T>));
    std::vector<double> ts;
    double tplan;
    std::string info;
    auto restore = [&] { fill(in.data(), n); };
    if (mode == "direct" || mode == "direct_inplace")
    {
        double t0 = now();
        dft_plan<T> p(n);
        tplan = now() - t0;
        univector<u8> tmp(p.temp_size);
        std::memset(tmp.data(), 0, tmp.size());
        for (int r = 0; r < reps; r++)
        {
            restore();
            double a = now();
            p.execute(mode == "direct" ? out.data() : in.data(), in.data(), tmp.data());
            ts.push_back(now() - a);
        }
    }
    else
    {
        double t0 = now();
        auto opt    = fs_opts<T>();
        opt.inplace = mode == "fs_inplace";
        fft_fourstep<T> f(n, opt);
        tplan = now() - t0;
        char s[96];
        snprintf(s, sizeof(s), " N1=%zu N2=%zu", f.n1(), f.n2());
        info = s;
        for (int r = 0; r < reps; r++)
        {
            restore();
            double a = now();
            if (mode == "fs")
                f.execute(out.data(), in.data());
            else
                f.execute_inplace(in.data());
            ts.push_back(now() - a);
        }
#if FS_PROFILE
        printf("  profile per run: s1.gather=%.2f s1.fft=%.2f s1.twiddle+store=%.2f | s2.gather=%.2f s2.fft=%.2f "
               "s2.scatter=%.2f | inplace: chunk=%.2f square=%.2f\n",
               f.prof[0] / reps, f.prof[8] / reps, f.prof[1] / reps, f.prof[2] / reps, f.prof[3] / reps,
               f.prof[4] / reps, f.prof[6] / reps, f.prof[7] / reps);
#endif
    }
    std::sort(ts.begin(), ts.end());
    printf("%-15s %s n=%zu plan=%.2fs exec min=%.3fs median=%.3fs peak_rss=%.3fGiB%s\n", mode.c_str(),
           sizeof(T) == 4 ? "f32" : "f64", n, tplan, ts[0], ts[ts.size() / 2], peak_gib(), info.c_str());
    return 0;
}

template <typename T>
static int check(size_t n)
{
    univector<complex<double>> ref(n);
    fill(ref.data(), n);
    univector<complex<T>> a0(n), a(n), o(n);
    for (size_t i = 0; i < n; i++) a0[i] = complex<T>(ref[i]);
    {
        dft_plan<double> p(n);
        univector<u8> t(p.temp_size);
        p.execute(ref.data(), ref.data(), t.data());
    }
    auto err = [&](const complex<T>* x)
    {
        double e = 0, s = 0;
        for (size_t i = 0; i < n; i++)
        {
            s += std::norm(ref[i]);
            e += std::norm(complex<double>(x[i]) - ref[i]);
        }
        return std::sqrt(e / s);
    };
    const double tol = sizeof(T) == 4 ? 1e-5 : 1e-12;
    int fails        = 0;
    auto report      = [&](const char* name, double e)
    {
        printf("  %-15s rel_rms=%.3g %s\n", name, e, e < tol ? "OK" : "FAIL");
        fails += !(e < tol);
    };
    printf("check %s n=%zu\n", sizeof(T) == 4 ? "f32" : "f64", n);
    {
        dft_plan<T> p(n);
        univector<u8> t(p.temp_size);
        p.execute(o.data(), a0.data(), t.data());
        report("direct", err(o.data()));
    }
    {
        fft_fourstep<T> f(n, fs_opts<T>());
        printf("  N1=%zu N2=%zu\n", f.n1(), f.n2());
        a = a0;
        f.execute(o.data(), a.data());
        report("fs", err(o.data()));
        bool same = std::memcmp((const void*)a.data(), (const void*)a0.data(), n * sizeof(complex<T>)) == 0;
        printf("  %-15s %s\n", "fs input kept", same ? "OK" : "FAIL");
        fails += !same;
        // inverse: back = idft(dft(x)) / n, computed out-of-place
        univector<complex<T>> back(n);
        f.execute(back.data(), o.data(), true);
        double e = 0, s = 0;
        for (size_t i = 0; i < n; i++)
        {
            s += std::norm(complex<double>(a0[i]));
            e += std::norm(complex<double>(back[i]) / double(n) - complex<double>(a0[i]));
        }
        report("fs_roundtrip", std::sqrt(e / s));
    }
    for (int restricted = 1; restricted >= 0; --restricted)
    {
        auto opt    = fs_opts<T>();
        opt.inplace = restricted;
        fft_fourstep<T> f(n, opt);
        printf("  inplace plan N1=%zu N2=%zu (%s)\n", f.n1(), f.n2(),
               f.inplace_without_work() ? "no work buffer" : "work buffer");
        a = a0;
        f.execute_inplace(a.data());
        report(restricted ? "fs_inplace" : "fs_inplace_work", err(a.data()));
        f.execute_inplace(a.data(), true);
        double e = 0, s = 0;
        for (size_t i = 0; i < n; i++)
        {
            s += std::norm(complex<double>(a0[i]));
            e += std::norm(complex<double>(a[i]) / double(n) - complex<double>(a0[i]));
        }
        report("inplace_roundtrip", std::sqrt(e / s));
    }
    return fails;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s N f32|f64 MODE|check [reps]\n", argv[0]);
        return 2;
    }
#ifdef __x86_64__
    if (const char* c = getenv("CPU"); c && *c)
    {
        std::string s = c;
        override_cpu(s == "sse2" ? cpu_t::sse2 : s == "sse41" ? cpu_t::sse41 : s == "avx" ? cpu_t::avx1 : cpu_t::avx2);
    }
#endif
    const size_t n = strtoull(argv[1], nullptr, 10);
    const bool f64 = !strcmp(argv[2], "f64");
    std::string mode = argv[3];
    if (mode == "check")
        return f64 ? check<double>(n) : check<float>(n);
    const int reps = argc > 4 ? std::max(1, atoi(argv[4])) : 3;
    return f64 ? timing<double>(n, mode, reps) : timing<float>(n, mode, reps);
}
