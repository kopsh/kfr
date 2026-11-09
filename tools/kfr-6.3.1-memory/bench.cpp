// Memory / speed / accuracy comparison: KFR dft_plan<T>(N) vs fft_fourstep<T>.
//
//   bench N f32|f64 MODE [reps]          time and peak RSS (input regenerated before every run, not timed)
//   bench N f32|f64 rdirect|rfs|rfs_inplace [reps]   real input, N real samples (KFR dft_plan_real vs four-step)
//   bench N f32|f64 check                accuracy of all modes vs f64 dft_plan, plus inverse, plus a 2-thread test
//   bench N f32|f64 ab A B [reps]        interleaved A/B timing in one process (A,B in turn, order alternates);
//                                        prints both medians and the median/min/max of the per-pair ratio B/A
// MODE: (append @N to a fs / fs_auto / fs_inplace mode in `ab` to run it with N threads)
// MODE: direct | direct_inplace | fs (out-of-place, input preserved) | fs_inplace (no work buffer)
//       | fs_inplace_work (in-place using an N-point work buffer) | fs_copy_inplace (memcpy + fs_inplace, input kept)
//       | fs_auto (execute() with default options: picks direct or copy + in-place)
// env:  CPU=sse2|sse41|avx|avx2 (x86 only, override KFR's runtime dispatch), FS_N1, FS_B, FS_PF (options)
#include "fourstep.hpp"
#include "fourstep_real.hpp"
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
        opt.oop_strategy   = mode == "fs" ? 0 : -1;
        fft_fourstep<T> f(n, opt);
        tplan = now() - t0;
        char s[96];
        snprintf(s, sizeof(s), " N1=%zu N2=%zu", f.n1(), f.n2());
        info = s;
        for (int r = 0; r < reps; r++)
        {
            restore();
            double a = now();
            if (mode == "fs" || mode == "fs_auto")
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
    if (mode == "rdirect" || mode == "rfs" || mode == "rfs_inplace")
    {
        // real input of n samples: the complex<T> buffers of the harness are used as 2n reals
        if (mode == "rdirect")
        {
            auto p = std::make_shared<dft_plan_real<T>>(n);
            auto t = std::make_shared<univector<u8>>(p->temp_size);
            r.run  = [=](complex<T>* in, complex<T>* out)
            { p->execute(out, reinterpret_cast<const T*>(in), t->data()); };
            return r;
        }
        auto opt = fs_opts<T>();
        auto f   = std::make_shared<fft_fourstep_real<T>>(n, mode == "rfs_inplace" ? fft_fourstep<T>::for_inplace() : opt);
        char s[96];
        snprintf(s, sizeof(s), "N1=%zu N2=%zu threads=%zu", f->core_plan().n1(), f->core_plan().n2(), threads);
        r.info = s;
        auto wss = std::make_shared<std::vector<typename fft_fourstep<T>::workspace>>(f->make_workspaces(threads));
        auto par = std::make_shared<typename fft_fourstep<T>::parallel_for_t>(fft_fourstep<T>::thread_executor(threads));
        if (mode == "rfs_inplace")
            r.run = [=](complex<T>* in, complex<T>*)
            {
                if (threads > 1)
                    f->execute_inplace(reinterpret_cast<T*>(in), *wss, *par);
                else
                    f->execute_inplace(reinterpret_cast<T*>(in), (*wss)[0]);
            };
        else
            r.run = [=](complex<T>* in, complex<T>* out)
            {
                if (threads > 1)
                    f->execute(out, reinterpret_cast<const T*>(in), *wss, *par);
                else
                    f->execute(out, reinterpret_cast<const T*>(in), (*wss)[0]);
            };
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

// ---------------------------------------------------------------- real input
template <typename T>
static void fill_real(T* a, size_t n)
{
    for (size_t i = 0; i < n; i++)
        a[i] = T(std::sin(0.001 * i) + ((i * 2654435761u) >> 28) * 0.01 + 0.5 * std::cos(0.37 * i));
}

// n real samples. Modes: rdirect (KFR dft_plan_real), rfs (four-step, input kept), rfs_inplace
template <typename T>
static int timing_real(size_t n, const std::string& mode, int reps)
{
    const size_t M = n / 2;
    const bool ip  = mode == "rfs_inplace";
    univector<T> in(ip ? n + 2 : n);
    univector<complex<T>> out(ip ? 0 : M + 1);
    std::vector<double> ts;
    double tplan;
    std::string info;
    if (mode == "rdirect")
    {
        double t0 = now();
        dft_plan_real<T> p(n);
        tplan = now() - t0;
        univector<u8> tmp(p.temp_size);
        for (int r = 0; r < reps; r++)
        {
            fill_real(in.data(), n);
            double a = now();
            p.execute(out.data(), in.data(), tmp.data());
            ts.push_back(now() - a);
        }
    }
    else
    {
        double t0 = now();
        fft_fourstep_real<T> f(n, ip ? fft_fourstep<T>::for_inplace() : fs_opts<T>());
        tplan = now() - t0;
        auto ws = f.make_workspace();
        char s[96];
        snprintf(s, sizeof(s), " N1=%zu N2=%zu", f.core_plan().n1(), f.core_plan().n2());
        info = s;
        for (int r = 0; r < reps; r++)
        {
            fill_real(in.data(), n);
            double a = now();
            if (ip)
                f.execute_inplace(in.data(), ws);
            else
                f.execute(out.data(), in.data(), ws);
            ts.push_back(now() - a);
        }
    }
    std::sort(ts.begin(), ts.end());
    printf("%-15s %s real n=%zu plan=%.2fs exec min=%.3fs median=%.3fs peak_rss=%.3fGiB%s\n", mode.c_str(),
           sizeof(T) == 4 ? "f32" : "f64", n, tplan, ts[0], ts[ts.size() / 2], peak_gib(), info.c_str());
    return 0;
}

template <typename T>
static int check_real(size_t n)
{
    const size_t M = n / 2;
    univector<double> x64(n);
    fill_real(x64.data(), n);
    univector<T> x(n + 2), x0(n + 2);
    for (size_t i = 0; i < n; i++) x0[i] = T(x64[i]);
    x0[n] = x0[n + 1] = 0;
    univector<complex<double>> ref(M + 1);
    {
        dft_plan_real<double> p(n);
        univector<u8> t(p.temp_size);
        p.execute(ref.data(), x64.data(), t.data());
    }
    auto err = [&](const complex<T>* o)
    {
        double e = 0, s = 0;
        for (size_t k = 0; k <= M; k++)
        {
            s += std::norm(ref[k]);
            e += std::norm(complex<double>(o[k]) - ref[k]);
        }
        return std::sqrt(e / s);
    };
    const double tol = sizeof(T) == 4 ? 1e-5 : 1e-12;
    int fails        = 0;
    auto report      = [&](const char* name, double e)
    {
        printf("  %-18s rel_rms=%.3g %s\n", name, e, e < tol ? "OK" : "FAIL");
        fails += !(e < tol);
    };
    printf("check real %s n=%zu\n", sizeof(T) == 4 ? "f32" : "f64", n);
    {
        dft_plan_real<T> p(n);
        univector<u8> t(p.temp_size);
        univector<complex<T>> o(M + 1);
        p.execute(o.data(), x0.data(), t.data());
        report("kfr dft_plan_real", err(o.data()));
    }
    fft_fourstep_real<T> f(n);
    printf("  core N1=%zu N2=%zu\n", f.core_plan().n1(), f.core_plan().n2());
    auto ws = f.make_workspace();
    univector<complex<T>> o(M + 1);
    f.execute(o.data(), x0.data(), ws);
    report("real fs", err(o.data()));
    // inverse, out-of-place, input kept: must give n * x
    auto inv_err = [&](const T* y)
    {
        double e = 0, s = 0;
        for (size_t i = 0; i < n; i++)
        {
            s += x64[i] * x64[i];
            e += (double(y[i]) / double(n) - x64[i]) * (double(y[i]) / double(n) - x64[i]);
        }
        return std::sqrt(e / s);
    };
    univector<T> back(n);
    univector<complex<T>> o_copy = o;
    f.inverse(back.data(), o.data(), ws);
    report("real inverse oop", inv_err(back.data()));
    const bool kept = std::memcmp((const void*)o.data(), (const void*)o_copy.data(), (M + 1) * sizeof(complex<T>)) == 0;
    printf("  %-18s %s\n", "inverse input kept", kept ? "OK" : "FAIL");
    fails += !kept;
    // in-place forward + inverse
    x = x0;
    f.execute_inplace(x.data(), ws);
    report("real fs inplace", err(reinterpret_cast<complex<T>*>(x.data())));
    f.inverse_inplace(x.data(), ws);
    report("real inv inplace", inv_err(x.data()));
    // multi-threaded: bit-identical to serial
    {
        auto par = fft_fourstep<T>::thread_executor(3);
        auto wss = f.make_workspaces(3);
        univector<complex<T>> o2(M + 1);
        f.execute(o2.data(), x0.data(), wss, par);
        bool same = std::memcmp((const void*)o.data(), (const void*)o2.data(), (M + 1) * sizeof(complex<T>)) == 0;
        univector<T> back2(n);
        f.inverse(back2.data(), o.data(), wss, par);
        same = same && std::memcmp((const void*)back.data(), (const void*)back2.data(), n * sizeof(T)) == 0;
        printf("  %-18s %s\n", "real mt identical", same ? "OK" : "FAIL");
        fails += !same;
    }
    return fails;
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
    if (getenv("FS_INFO"))
        printf("l2_cache_bytes=%zu\n", fft_fourstep<float>::l2_cache_bytes());
    const size_t n = strtoull(argv[1], nullptr, 10);
    const bool f64 = !strcmp(argv[2], "f64");
    std::string mode = argv[3];
    if (mode == "check")
        return (f64 ? check<double>(n) : check<float>(n)) + (n % 2 == 0 && fft_fourstep_real<float>::find_split(n) ? (f64 ? check_real<double>(n) : check_real<float>(n)) : 0);
    if (mode == "rdirect" || mode == "rfs" || mode == "rfs_inplace")
        return f64 ? timing_real<double>(n, mode, argc > 4 ? std::max(1, atoi(argv[4])) : 3)
                   : timing_real<float>(n, mode, argc > 4 ? std::max(1, atoi(argv[4])) : 3);
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
