// Memory / speed / accuracy comparison: KFR dft_plan<T>(N) vs fft_fourstep<T>.
//
//   bench N f32|f64 MODE [reps]          time and peak RSS (input regenerated before every run, not timed)
//   bench N f32|f64 check                accuracy of all modes vs f64 dft_plan, plus inverse, plus a 2-thread test
//   bench N f32|f64 ab A B [reps]        interleaved A/B timing in one process (A,B in turn, order alternates);
//                                        prints both medians and the median/min/max of the per-pair ratio B/A
// MODE: (append @N to a fs / fs_auto / fs_inplace mode in `ab` to run it with N threads)
// MODE: direct | direct_inplace | fs (out-of-place, input preserved) | fs_inplace (no work buffer)
//       | fs_inplace_work (in-place using an N-point work buffer) | fs_copy_inplace (memcpy + fs_inplace, input kept)
//       | fs_auto (execute() with default options: picks direct or copy + in-place)
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
#include <functional>
#include <string>
#include <thread>

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
        auto opt           = fs_opts<T>();
        opt.prefer_inplace = mode != "fs" && mode != "fs_inplace_work";
        opt.inplace_slack  = mode == "fs_inplace" ? 1e9 : opt.inplace_slack;
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
    }
    std::sort(ts.begin(), ts.end());
    printf("%-15s %s n=%zu plan=%.2fs exec min=%.3fs median=%.3fs peak_rss=%.3fGiB%s\n", mode.c_str(),
           sizeof(T) == 4 ? "f32" : "f64", n, tplan, ts[0], ts[ts.size() / 2], peak_gib(), info.c_str());
    return 0;
}

// ---------------------------------------------------------------- interleaved A/B timing
template <typename T>
struct runner
{
    std::function<void(complex<T>* in, complex<T>* out)> run;
    std::string info;
};

template <typename T>
static runner<T> make_runner(size_t n, std::string mode)
{
    runner<T> r;
    size_t threads = 1; // "fs_inplace@4": run the four-step modes with 4 threads
    if (auto at = mode.find('@'); at != std::string::npos)
    {
        threads = std::max(1, atoi(mode.c_str() + at + 1));
        mode.resize(at);
    }
    if (mode == "direct" || mode == "direct_inplace")
    {
        auto p  = std::make_shared<dft_plan<T>>(n);
        auto t  = std::make_shared<univector<u8>>(p->temp_size);
        bool ip = mode == "direct_inplace";
        r.run   = [=](complex<T>* in, complex<T>* out) { p->execute(ip ? in : out, in, t->data()); };
        return r;
    }
    auto opt           = fs_opts<T>();
    opt.prefer_inplace = mode != "fs";
    opt.inplace_slack  = mode == "fs" || mode == "fs_auto" ? opt.inplace_slack : 1e9;
    opt.oop_strategy   = mode == "fs" ? 0 : mode == "fs_copy_inplace" ? 1 : -1;
    auto f             = std::make_shared<fft_fourstep<T>>(n, opt);
    auto ws            = std::make_shared<typename fft_fourstep<T>::workspace>(f->make_workspace());
    char s[96];
    snprintf(s, sizeof(s), "N1=%zu N2=%zu threads=%zu", f->n1(), f->n2(), threads);
    r.info = s;
    if (threads > 1)
    {
        auto wss = std::make_shared<std::vector<typename fft_fourstep<T>::workspace>>(f->make_workspaces(threads));
        auto par = std::make_shared<typename fft_fourstep<T>::parallel_for_t>(fft_fourstep<T>::thread_executor(threads));
        if (mode == "fs" || mode == "fs_auto")
            r.run = [=](complex<T>* in, complex<T>* out) { f->execute(out, in, false, *wss, *par); };
        else if (mode == "fs_inplace")
            r.run = [=](complex<T>* in, complex<T>*) { f->execute_inplace(in, false, *wss, *par); };
        else
            throw std::invalid_argument("mode " + mode + " has no multi-threaded variant");
        return r;
    }
    if (mode == "fs" || mode == "fs_auto")
        r.run = [=](complex<T>* in, complex<T>* out) { f->execute(out, in, false, *ws); };
    else if (mode == "fs_inplace")
        r.run = [=](complex<T>* in, complex<T>*) { f->execute_inplace(in, false, *ws); };
    else if (mode == "fs_copy_inplace")
        r.run = [=](complex<T>* in, complex<T>* out) { f->execute_copy_inplace(out, in, false, *ws); };
    else
        throw std::invalid_argument("unknown mode " + mode);
    return r;
}

template <typename T>
static int ab(size_t n, const std::string& ma, const std::string& mb, int reps)
{
    univector<complex<T>> src(n), in(n), out(n);
    fill(src.data(), n);
    std::memset((void*)out.data(), 0, n * sizeof(complex<T>));
    runner<T> A = make_runner<T>(n, ma), B = make_runner<T>(n, mb);
    std::vector<double> ta, tb, ratio;
    auto timed = [&](runner<T>& r)
    {
        std::memcpy((void*)in.data(), src.data(), n * sizeof(complex<T>));
        const double t0 = now();
        r.run(in.data(), out.data());
        return now() - t0;
    };
    for (int i = 0; i < reps; ++i)
    {
        double a, b;
        if (i % 2 == 0)
        {
            a = timed(A);
            b = timed(B);
        }
        else
        {
            b = timed(B);
            a = timed(A);
        }
        ta.push_back(a);
        tb.push_back(b);
        ratio.push_back(b / a);
    }
    auto med = [](std::vector<double> v)
    {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto mn = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
    auto mx = [](const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); };
    printf("%s n=%zu reps=%d\n  A %-16s median %.3fs min %.3fs %s\n  B %-16s median %.3fs min %.3fs %s\n"
           "  ratio B/A: median %.3f  (min %.3f, max %.3f)\n",
           sizeof(T) == 4 ? "f32" : "f64", n, reps, ma.c_str(), med(ta), mn(ta), A.info.c_str(), mb.c_str(), med(tb),
           mn(tb), B.info.c_str(), med(ratio), mn(ratio), mx(ratio));
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
    {
        // two threads share one plan, each with its own workspace
        fft_fourstep<T> f(n, fs_opts<T>());
        auto w1 = f.make_workspace(), w2 = f.make_workspace();
        univector<complex<T>> o1(n), o2(n);
        std::thread t1([&] { f.execute(o1.data(), a0.data(), false, w1); });
        std::thread t2([&] { f.execute(o2.data(), a0.data(), false, w2); });
        t1.join();
        t2.join();
        const bool same = std::memcmp((const void*)o1.data(), (const void*)o2.data(), n * sizeof(complex<T>)) == 0;
        report("2 threads", err(o1.data()));
        printf("  %-15s %s\n", "2 threads equal", same ? "OK" : "FAIL");
        fails += !same;
    }
    for (int ip = 0; ip < 2; ++ip)
    {
        // opt-in multi-threading: results must be bit-identical to the serial ones
        auto opt = fs_opts<T>();
        opt.oop_strategy = ip ? 1 : 0;
        fft_fourstep<T> f(n, ip ? fft_fourstep<T>::for_inplace() : opt);
        auto par = fft_fourstep<T>::thread_executor(3);
        auto wss = f.make_workspaces(3);
        univector<complex<T>> serial(n), parallel(n);
        auto w1 = f.make_workspace();
        for (int inv = 0; inv < 2; ++inv)
        {
            if (ip)
            {
                serial = a0;
                parallel = a0;
                f.execute_inplace(serial.data(), inv, w1);
                f.execute_inplace(parallel.data(), inv, wss, par);
            }
            else
            {
                f.execute(serial.data(), a0.data(), inv, w1);
                f.execute(parallel.data(), a0.data(), inv, wss, par);
            }
            const bool same = std::memcmp((const void*)serial.data(), (const void*)parallel.data(), n * sizeof(complex<T>)) == 0;
            printf("  %-15s %s\n", ip ? (inv ? "mt inplace inv" : "mt inplace") : (inv ? "mt oop inv" : "mt oop"),
                   same ? "OK (bit-identical)" : "FAIL");
            fails += !same;
        }
    }
    {
        fft_fourstep<T> f(n); // default options
        a = a0;
        f.execute_inplace(a.data());
        printf("  default plan N1=%zu N2=%zu (%s)\n", f.n1(), f.n2(),
               f.inplace_without_work() ? "in-place without work buffer" : "in-place needs temporary buffer");
        report("fs_default", err(a.data()));
    }
    for (int restricted = 1; restricted >= 0; --restricted)
    {
        auto opt           = fs_opts<T>();
        opt.prefer_inplace = restricted;
        opt.inplace_slack  = 1e9;
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
    if (mode == "ab")
    {
        if (argc < 6)
        {
            fprintf(stderr, "usage: %s N f32|f64 ab MODE_A MODE_B [reps]\n", argv[0]);
            return 2;
        }
        const int r = argc > 6 ? std::max(2, atoi(argv[6])) : 6;
        return f64 ? ab<double>(n, argv[4], argv[5], r) : ab<float>(n, argv[4], argv[5], r);
    }
    const int reps = argc > 4 ? std::max(1, atoi(argv[4])) : 3;
    return f64 ? timing<double>(n, mode, reps) : timing<float>(n, mode, reps);
}
