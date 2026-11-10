// Four-step FFT for very large sizes, built on KFR 6.3.1's public dft_plan.
//
//   N = N1 * N2,  x[n1*N2 + n2]  ->  X[k1 + N1*k2]
//   step 1: for every column n2: length-N1 DFT, multiply by w_N^(n2*k1)
//   step 2: for every row    k1: length-N2 DFT
//
// Extra memory is a few MB (two small KFR plans, sqrt(N)-sized twiddle tables, a block buffer)
// instead of ~2N complex for dft_plan<T>(N) (twiddles + scratch).
//
// Two execution modes, both with natural-order output and unnormalized inverse (same as dft_plan):
//
//   execute(out, in)       out-of-place, `in` is preserved.            memory: in + out
//   execute_inplace(x)     in-place.                                   memory: x only
//                          Needs a split where N1 divides N2 or N2 divides N1. The default options prefer
//                          such a split (unless it is more than options.inplace_slack slower, measured);
//                          without one it falls back to a temporary N-point buffer for that call.
//
// Thread safety: the plan itself (sub-plans, twiddle tables) is immutable. The per-call scratch memory
// (a few MB) lives in a `workspace`. The overloads that take a workspace are const and may be called
// concurrently with one workspace per thread; the overloads without it use one internal workspace and
// must not be called concurrently.
//
// Out-of-place data flow (no transpose pass, no extra buffer):
//   step 1 gathers B columns of `in` (strided), transforms them and writes them contiguously to out as
//          out[n2*N1 + k1];
//   step 2 gathers B columns of `out` (strided), transforms them and writes them back to the same
//          locations, which is exactly X[k1 + N1*k2].
//
// In-place data flow: the same two steps on the [k1][n2] layout (every block is written back to where it
// was read from), then an in-place transpose of the N1 x N2 matrix that moves whole rows (chunks) and
// transposes square blocks, so it needs only one row of scratch.
#pragma once
#include <kfr/dft.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <cstdio>
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

// KFR handles the length without Bluestein (no prime factor left > 10 after radices 2..10)
inline bool fs_kfr_fast(size_t n)
{
    for (size_t r = 10; r >= 2; --r)
        while (n % r == 0) n /= r;
    return n < 101;
}

// Length-L DFT with the interface of kfr::dft_plan. Lengths KFR handles directly use dft_plan. Lengths with a
// prime factor >= 101 (KFR would run Bluestein with a power-of-two convolution of length >= 2L-1) use Bluestein
// here with the convolution length rounded up to the nearest 2-3-5-7-smooth number instead, which is up to
// 1.4x shorter than the next power of two. In-place use (out == in) is supported.
template <typename T>
class fs_sub_dft
{
public:
    using C = kfr::complex<T>;
    size_t temp_size = 0, data_size = 0;

    fs_sub_dft() = default;
    explicit fs_sub_dft(size_t len, size_t conv_len = 0) : L(len)
    {
        if (fs_kfr_fast(len) || len < 2)
        {
            plan      = kfr::dft_plan<T>(len);
            temp_size = plan.temp_size;
            data_size = plan.data_size;
            return;
        }
        blue = true;
        M    = conv_len ? conv_len : smooth_at_least(2 * len - 1);
        plan = kfr::dft_plan<T>(M);
        chirp.resize(2 * len);
        for (size_t n = 0; n < len; ++n) // w_n = exp(-i*pi*n^2/L), n^2 reduced mod 2L exactly
        {
            const unsigned long long r = (unsigned long long)n * n % (2ULL * len);
            const long double a        = -3.141592653589793238462643383279502884L * (long double)r / (long double)len;
            chirp[2 * n]     = T(std::cos(a));
            chirp[2 * n + 1] = T(std::sin(a));
        }
        // spectrum of conj(w) wrapped to length M, scaled by 1/M (the inverse FFT below is unnormalized)
        kfr::univector<C> b(M, C(T(0), T(0)));
        for (size_t n = 0; n < len; ++n)
        {
            const C v(chirp[2 * n], -chirp[2 * n + 1]);
            b[n] = v;
            if (n) b[M - n] = v;
        }
        kfr::univector<kfr::u8> tmp(plan.temp_size);
        plan.execute(b.data(), b.data(), tmp.data());
        bhat.resize(M);
        for (size_t i = 0; i < M; ++i) bhat[i] = b[i] * T(1.0L / (long double)M);
        temp_size = M * sizeof(C) + plan.temp_size;
        data_size = plan.data_size + (chirp.size() + bhat.size() * 2) * sizeof(T);
    }
    size_t conv_len() const { return blue ? M : L; }
    bool bluestein() const { return blue; }

    template <bool inv>
    void execute(C* out, const C* in, kfr::u8* temp, kfr::cbool_t<inv> c) const
    {
        if (!blue)
        {
            plan.execute(out, in, temp, c);
            return;
        }
        C* buf        = reinterpret_cast<C*>(temp);
        kfr::u8* ptmp = temp + M * sizeof(C);
        const C* w    = reinterpret_cast<const C*>(chirp.data());
        // inverse DFT(x) = conj(DFT(conj x))
        for (size_t n = 0; n < L; ++n)
        {
            const C v = inv ? C(in[n].real(), -in[n].imag()) : in[n];
            buf[n]    = v * w[n];
        }
        std::fill(buf + L, buf + M, C(T(0), T(0)));
        plan.execute(buf, buf, ptmp);
        for (size_t i = 0; i < M; ++i) buf[i] = buf[i] * bhat[i];
        plan.execute(buf, buf, ptmp, kfr::ctrue);
        for (size_t k = 0; k < L; ++k)
        {
            const C v = buf[k] * w[k];
            out[k]    = inv ? C(v.real(), -v.imag()) : v;
        }
    }

    static size_t smooth_at_least(size_t n)
    {
        size_t best = size_t(1) << 62;
        for (size_t p2 = 1; p2 < best; p2 *= 2)
            for (size_t p3 = p2; p3 < best; p3 *= 3)
                for (size_t p5 = p3; p5 < best; p5 *= 5)
                    for (size_t p7 = p5; p7 < best; p7 *= 7)
                        if (p7 >= n) { best = std::min(best, p7); break; }
        return best;
    }

private:
    size_t L = 0, M = 0;
    bool blue = false;
    kfr::dft_plan<T> plan;
    std::vector<T> chirp;
    kfr::univector<C> bhat;
};

template <typename T>
class fft_fourstep
{
public:
    using C = kfr::complex<T>;

    struct options
    {
        size_t n1         = 0;    // force N1 (0 = choose automatically)
        bool measure      = true; // time candidate sub-FFT lengths on this machine
        bool prefer_inplace    = true; // prefer a split that lets execute_inplace() run without a work buffer
        double inplace_slack   = 1.05; // ... unless its measured cost is more than this factor above the best split
        int oop_strategy       = -1;   // execute(): -1 auto, 0 always direct, 1 always copy + in-place
        size_t min_len    = 512;
        size_t max_bytes  = 0;    // max sub-FFT size in bytes (0 = auto: half of the L2 cache, between 128 KiB and 1 MiB)
        size_t block      = 16;   // columns/rows per block
        size_t prefetch   = 8;    // prefetch distance in rows (0 = off)
    };

    /// Options for callers that only use execute_inplace(): always pick a split that needs no work buffer.
    static options for_inplace()
    {
        options o;
        o.inplace_slack = 1e9;
        return o;
    }

    explicit fft_fourstep(size_t n) : fft_fourstep(n, options{}) {}
    fft_fourstep(size_t n, const options& opt) : N(n), B(opt.block), pf(opt.prefetch), oop_strategy(opt.oop_strategy)
    {
        N1 = opt.n1 ? opt.n1 : choose_n1(n, opt);
        if (N1 == 0 || n % N1 != 0)
            throw std::invalid_argument("fft_fourstep: no usable N1 * N2 split");
        N2 = n / N1;
        p1 = fs_sub_dft<T>(N1);
        p2 = fs_sub_dft<T>(N2);
        while ((size_t(1) << (2 * Lbits)) < N) ++Lbits;
        const size_t L = size_t(1) << Lbits;
        lo.resize(2 * L);
        hi.resize(2 * (N / L + 1));
        for (size_t r = 0; r < L; ++r) cexp_m(r, lo.data() + 2 * r);
        for (size_t q = 0; q < N / L + 1; ++q) cexp_m(q * L, hi.data() + 2 * q);
        wj.resize(2 * N1 * B); // wj[k1][j] = w_N^(j*k1)
        for (size_t k1 = 0; k1 < N1; ++k1)
            for (size_t j = 0; j < B; ++j) cexp_m(j * k1 % N, wj.data() + 2 * (k1 * B + j));
    }

    /// Size of one core's L2 cache in bytes, or 0 if it cannot be determined.
    static size_t l2_cache_bytes()
    {
#if defined(_SC_LEVEL2_CACHE_SIZE)
        if (long v = sysconf(_SC_LEVEL2_CACHE_SIZE); v > 0)
            return size_t(v);
#endif
        for (int idx = 0; idx < 8; ++idx) // Linux sysfs: cpu0/cache/index<i>/{level,size}
        {
            char path[96], buf[32] = {};
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
            FILE* f = fopen(path, "r");
            if (!f)
                break;
            const bool l2 = fgets(buf, sizeof(buf), f) && atoi(buf) == 2;
            fclose(f);
            if (!l2)
                continue;
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
            if (!(f = fopen(path, "r")))
                break;
            size_t bytes = 0;
            if (fgets(buf, sizeof(buf), f))
            {
                bytes = size_t(atol(buf));
                if (strchr(buf, 'K'))
                    bytes <<= 10;
                else if (strchr(buf, 'M'))
                    bytes <<= 20;
            }
            fclose(f);
            return bytes;
        }
        return 0;
    }

    /// Returns a usable split for n, or 0 when there is none (fall back to dft_plan<T>(n) then).
    static size_t find_split(size_t n, bool inplace = false)
    {
        options o;
        o.measure        = false;
        o.prefer_inplace = inplace;
        return choose_n1(n, o);
    }

    /// Per-call scratch memory, one per thread for concurrent calls. In-place execution needs
    /// B * N1 complex numbers plus the temp of the sub-plans (about 1 MB); the direct out-of-place path grows
    /// the buffer to B * max(N1, N2) complex numbers on first use.
    struct workspace
    {
        kfr::univector<kfr::u8> temp;
        kfr::univector<C> buf;
        size_t bytes() const { return temp.size() + buf.size() * sizeof(C); }
    };
    workspace make_workspace() const
    {
        workspace w;
        w.temp.resize(std::max(p1.temp_size, p2.temp_size));
        w.buf.resize(B * N1);
        return w;
    }

    struct tune_result
    {
        options best;                                   ///< `base` with n1 set to the fastest split found
        std::vector<std::pair<size_t, double>> tried;   ///< (N1, best seconds) for every candidate that was run
    };

    /// Opt-in tuning on the real size: runs the `max_candidates` most promising splits (ranked by the sub-FFT
    /// timing model) a few times and returns the options of the fastest one. Use it once per size on the
    /// target machine and keep the resulting options.n1. Needs memory for the data it transforms:
    /// n complex numbers for `inplace` mode, 2n for out-of-place.
    static tune_result autotune(size_t n, const options& base = options{}, bool inplace = true, int reps = 2,
                                size_t max_candidates = 6)
    {
        options o = base;
        std::vector<size_t> cand = all_splits(n, o);
        if (inplace)
        {
            std::vector<size_t> d;
            for (size_t x : cand)
                if (divisible_split(n, x))
                    d.push_back(x);
            if (!d.empty())
                cand = d;
        }
        std::map<size_t, double> quick;
        for (size_t d : cand) quick[d] = cost_per_point(d, 1) + cost_per_point(n / d, 1);
        std::sort(cand.begin(), cand.end(), [&](size_t a, size_t b) { return quick[a] < quick[b]; });
        if (cand.size() > max_candidates)
            cand.resize(max_candidates);
        tune_result res;
        res.best = base;
        kfr::univector<C> a(n), out(inplace ? 0 : n);
        double best = 1e30;
        for (size_t d : cand)
        {
            options t        = base;
            t.n1             = d;
            t.prefer_inplace = inplace;
            if (inplace)
                t.inplace_slack = 1e9;
            t.oop_strategy = inplace ? t.oop_strategy : 0;
            fft_fourstep f(n, t);
            auto ws       = f.make_workspace();
            double fastest = 1e30;
            for (int r = 0; r < std::max(1, reps); ++r)
            {
                for (size_t i = 0; i < n; ++i) // cheap refill: the unnormalized transform would overflow otherwise
                    a[i] = C(T(i & 1023) * T(1e-3), T((i >> 10) & 1023) * T(1e-3));
                const double t0 = now();
                if (inplace)
                    f.execute_inplace(a.data(), false, ws);
                else
                    f.execute(out.data(), a.data(), false, ws);
                fastest = std::min(fastest, now() - t0);
            }
            res.tried.emplace_back(d, fastest);
            if (fastest < best)
            {
                best        = fastest;
                res.best.n1 = d;
            }
        }
        return res;
    }

    size_t size() const { return N; }
    size_t n1() const { return N1; }
    size_t n2() const { return N2; }
    /// True if execute_inplace() needs no N-point buffer for this plan.
    bool inplace_without_work() const { return N2 % N1 == 0 || N1 % N2 == 0; }
    /// Memory owned by the plan (twiddles, sub-plans), not counting workspaces.
    size_t plan_bytes() const
    {
        return p1.data_size + p2.data_size + (lo.size() + hi.size() + wj.size()) * sizeof(T);
    }
    /// Bytes of one workspace for in-place execution (also copy + in-place) and for the direct out-of-place path.
    size_t workspace_bytes_inplace() const { return std::max(p1.temp_size, p2.temp_size) + B * N1 * sizeof(C); }
    size_t workspace_bytes_oop() const
    {
        return std::max(p1.temp_size, p2.temp_size) + B * std::max(N1, N2) * sizeof(C);
    }

    /// True if execute() runs as copy + in-place transform. Measured: with a power-of-two N1 the strided
    /// column accesses of the direct out-of-place algorithm alias in the cache and copy + in-place is faster.
    bool oop_via_inplace() const
    {
        if (!inplace_without_work())
            return false;
        if (oop_strategy >= 0)
            return oop_strategy == 1;
        return (N1 & (N1 - 1)) == 0;
    }

    // ---- single-threaded API (one workspace)

    /// Out-of-place, `in` is preserved. `out` must not overlap `in`. Thread-safe with one workspace per thread.
    void execute(C* out, const C* in, bool inverse, workspace& ws) const { exec(out, in, inverse, &ws, nullptr); }

    /// In-place. No extra N-point memory if inplace_without_work(); otherwise a temporary N-point buffer.
    void execute_inplace(C* x, bool inverse, workspace& ws) const { exec_inplace(x, inverse, &ws, nullptr); }

    /// Out-of-place with the input preserved, as copy + in-place transform (same memory as execute()).
    void execute_copy_inplace(C* out, const C* in, bool inverse, workspace& ws) const
    {
        copy_inplace(out, in, inverse, &ws, nullptr);
    }

    // ---- multi-threaded API (opt-in)
    //
    // `par(count, body)` must call body(item, worker) for every item in [0, count), in any order and possibly
    // concurrently, and return when all are done. `worker` is in [0, wss.size()) and no two concurrent calls
    // may share a worker id. Items are independent blocks of work (columns, rows, tile rows), so this maps
    // directly onto a thread pool, OpenMP or a task system. The extra memory is one workspace per worker.
    using parallel_for_t = std::function<void(size_t count, const std::function<void(size_t item, size_t worker)>&)>;

    /// Simple executor that starts `threads - 1` std::threads for every parallel phase (a few per call).
    static parallel_for_t thread_executor(size_t threads)
    {
        return [threads](size_t count, const std::function<void(size_t, size_t)>& body)
        {
            std::atomic<size_t> next{ 0 };
            auto work = [&](size_t w)
            {
                for (;;)
                {
                    const size_t i = next.fetch_add(1);
                    if (i >= count)
                        break;
                    body(i, w);
                }
            };
            std::vector<std::thread> pool;
            for (size_t w = 1; w < std::min(threads, count); ++w) pool.emplace_back(work, w);
            work(0);
            for (auto& t : pool) t.join();
        };
    }
    std::vector<workspace> make_workspaces(size_t count) const
    {
        std::vector<workspace> v;
        for (size_t i = 0; i < count; ++i) v.push_back(make_workspace());
        return v;
    }

    void execute(C* out, const C* in, bool inverse, std::vector<workspace>& wss, const parallel_for_t& par) const
    {
        exec(out, in, inverse, wss.data(), &par);
    }
    void execute_inplace(C* x, bool inverse, std::vector<workspace>& wss, const parallel_for_t& par) const
    {
        exec_inplace(x, inverse, wss.data(), &par);
    }

    /// Pointer form of the API above: `wss` points to one workspace per worker, `par` may be null (serial).
    void execute_inplace(C* x, bool inverse, workspace* wss, const parallel_for_t* par) const
    {
        exec_inplace(x, inverse, wss, par);
    }

    // ---- convenience overloads using one internal workspace (not thread-safe)
    void execute(C* out, const C* in, bool inverse = false) { execute(out, in, inverse, own_ws()); }
    void execute_inplace(C* x, bool inverse = false) { execute_inplace(x, inverse, own_ws()); }

private:
    using body_t = std::function<void(size_t, size_t)>;

    template <class F>
    static void for_each(size_t count, const parallel_for_t* par, F&& f)
    {
        if (par && *par && count > 1)
            (*par)(count, body_t(std::ref(f)));
        else
            for (size_t i = 0; i < count; ++i) f(i, size_t(0));
    }

    static double now()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    static void pf_read(const void* p, size_t bytes)
    {
        for (size_t q = 0; q < bytes; q += 64) __builtin_prefetch(static_cast<const char*>(p) + q, 0);
    }
    static void pf_write(void* p, size_t bytes)
    {
        for (size_t q = 0; q < bytes; q += 64) __builtin_prefetch(static_cast<char*>(p) + q, 1);
    }

    void exec(C* out, const C* in, bool inverse, workspace* wss, const parallel_for_t* par) const
    {
        if (oop_via_inplace())
            copy_inplace(out, in, inverse, wss, par);
        else
            oop_direct(out, in, inverse, wss, par);
    }
    void oop_direct(C* out, const C* in, bool inverse, workspace* wss, const parallel_for_t* par) const
    {
        if (inverse)
            run_oop<true>(out, in, wss, par);
        else
            run_oop<false>(out, in, wss, par);
    }
    void copy_inplace(C* out, const C* in, bool inverse, workspace* wss, const parallel_for_t* par) const
    {
        const size_t parts = par ? 64 : 1, per = (N + parts - 1) / parts;
        for_each(parts, par,
                 [&](size_t i, size_t)
                 {
                     const size_t a = i * per, e = std::min(N, a + per);
                     if (a < e)
                         std::memcpy(static_cast<void*>(out + a), in + a, (e - a) * sizeof(C));
                 });
        exec_inplace(out, inverse, wss, par);
    }
    void exec_inplace(C* x, bool inverse, workspace* wss, const parallel_for_t* par) const
    {
        if (inplace_without_work())
        {
            if (inverse)
                run_inplace<true>(x, wss, par);
            else
                run_inplace<false>(x, wss, par);
        }
        else
        {
            std::unique_ptr<C[]> tmp(new C[N]);
            oop_direct(tmp.get(), x, inverse, wss, par);
            std::memcpy(static_cast<void*>(x), tmp.get(), N * sizeof(C));
        }
    }

    // Step 1 on B columns: gather x[n1][c0..c0+nb) (row stride N2 complex) and run the length-N1 DFTs.
    // The result is left in ws.buf as b[j*N1 + k1].
    template <bool inv>
    void step1_block(const T* x, size_t c0, size_t nb, workspace& ws) const
    {
        T* b       = reinterpret_cast<T*>(ws.buf.data());
        kfr::u8* t = ws.temp.data();
        for (size_t n1 = 0; n1 < N1; ++n1)
        {
            const T* row = x + 2 * (n1 * N2 + c0);
            if (pf && n1 + pf < N1)
                pf_read(x + 2 * ((n1 + pf) * N2 + c0), nb * 2 * sizeof(T));
            T* d = b + 2 * n1;
            for (size_t j = 0; j < nb; ++j)
            {
                d[2 * j * N1]     = row[2 * j];
                d[2 * j * N1 + 1] = row[2 * j + 1];
            }
        }
        for (size_t j = 0; j < nb; ++j)
        {
            C* col = ws.buf.data() + j * N1;
            p1.execute(col, col, t, kfr::cbool<inv>);
        }
    }

    // Twiddle w_N^((c0+j)*k1) of the block and store. Result of step 1 for column c0+j, frequency k1 is
    // d[j*dj + k1*dk] (out-of-place: dj = N1, dk = 1; in-place: dj = 1, dk = N2).
    template <bool inv>
    void step1_store(T* d0, size_t dj, size_t dk, size_t c0, size_t nb, const workspace& ws) const
    {
        const T* b        = reinterpret_cast<const T*>(ws.buf.data());
        const size_t mask = (size_t(1) << Lbits) - 1;
        size_t m          = 0; // c0 * k1
        for (size_t k1 = 0; k1 < N1; ++k1, m += c0)
        {
            const T* L0 = lo.data() + 2 * (m & mask);
            const T* H0 = hi.data() + 2 * (m >> Lbits);
            const T sr  = L0[0] * H0[0] - L0[1] * H0[1];
            const T si  = L0[0] * H0[1] + L0[1] * H0[0];
            const T* tw = wj.data() + 2 * k1 * B;
            const T* s  = b + 2 * k1;
            T* d        = d0 + 2 * k1 * dk;
            if (dk != 1 && pf && k1 + pf < N1)
                pf_write(d0 + 2 * (k1 + pf) * dk, nb * 2 * sizeof(T));
            for (size_t j = 0; j < nb; ++j)
            {
                const T wr = sr * tw[2 * j] - si * tw[2 * j + 1];
                const T wi = inv ? -(sr * tw[2 * j + 1] + si * tw[2 * j]) : sr * tw[2 * j + 1] + si * tw[2 * j];
                const T dr = s[2 * j * N1];
                const T di = s[2 * j * N1 + 1];
                d[2 * j * dj]     = dr * wr - di * wi;
                d[2 * j * dj + 1] = dr * wi + di * wr;
            }
        }
    }

    template <bool inv>
    void run_oop(C* dst, const C* src, workspace* wss, const parallel_for_t* par) const
    {
        const T* x = reinterpret_cast<const T*>(src);
        T* o       = reinterpret_cast<T*>(dst);
        // step 1: columns of `in`, results stored contiguously as o[n2*N1 + k1]
        for_each((N2 + B - 1) / B, par,
                 [&](size_t blk, size_t w)
                 {
                     workspace& ws   = wss[w];
                     const size_t c0 = blk * B, nb = std::min(B, N2 - c0);
                     step1_block<inv>(x, c0, nb, ws);
                     step1_store<inv>(o + 2 * (c0 * N1), N1, 1, c0, nb, ws);
                 });
        // step 2: columns of o (stride N1), transformed in place
        for_each((N1 + B - 1) / B, par,
                 [&](size_t blk, size_t w)
                 {
                     workspace& ws = wss[w];
                     if (ws.buf.size() < B * N2)
                         ws.buf.resize(B * N2); // grown once per workspace; step 1 of this call is complete
                     T* b            = reinterpret_cast<T*>(ws.buf.data());
                     const size_t r0 = blk * B, nb = std::min(B, N1 - r0);
                     for (size_t n2 = 0; n2 < N2; ++n2)
                     {
                         const T* seg = o + 2 * (n2 * N1 + r0);
                         if (pf && n2 + pf < N2)
                             pf_read(o + 2 * ((n2 + pf) * N1 + r0), nb * 2 * sizeof(T));
                         T* d = b + 2 * n2;
                         for (size_t j = 0; j < nb; ++j)
                         {
                             d[2 * j * N2]     = seg[2 * j];
                             d[2 * j * N2 + 1] = seg[2 * j + 1];
                         }
                     }
                     for (size_t j = 0; j < nb; ++j)
                     {
                         C* col = ws.buf.data() + j * N2;
                         p2.execute(col, col, ws.temp.data(), kfr::cbool<inv>);
                     }
                     for (size_t k2 = 0; k2 < N2; ++k2)
                     {
                         T* d = o + 2 * (k2 * N1 + r0);
                         if (pf && k2 + pf < N2)
                             pf_write(o + 2 * ((k2 + pf) * N1 + r0), nb * 2 * sizeof(T));
                         const T* s = b + 2 * k2;
                         for (size_t j = 0; j < nb; ++j)
                         {
                             d[2 * j]     = s[2 * j * N2];
                             d[2 * j + 1] = s[2 * j * N2 + 1];
                         }
                     }
                 });
    }

    template <bool inv>
    void run_inplace(C* xc, workspace* wss, const parallel_for_t* par) const
    {
        T* x = reinterpret_cast<T*>(xc);
        // step 1 on the [n1][n2] layout, written back to the locations it was read from
        for_each((N2 + B - 1) / B, par,
                 [&](size_t blk, size_t w)
                 {
                     workspace& ws   = wss[w];
                     const size_t c0 = blk * B, nb = std::min(B, N2 - c0);
                     step1_block<inv>(x, c0, nb, ws);
                     step1_store<inv>(x + 2 * c0, 1, N2, c0, nb, ws);
                 });
        // step 2: rows are contiguous, transform in place
        for_each((N1 + B - 1) / B, par,
                 [&](size_t blk, size_t w)
                 {
                     const size_t r0 = blk * B, r1 = std::min(N1, r0 + B);
                     for (size_t k1 = r0; k1 < r1; ++k1)
                     {
                         if (pf && k1 + 1 < N1)
                             pf_read(x + 2 * ((k1 + 1) * N2), 4096);
                         C* row = xc + k1 * N2;
                         p2.execute(row, row, wss[w].temp.data(), kfr::cbool<inv>);
                     }
                 });
        transpose_inplace(xc, N1, N2, par);
    }

    // ---- in-place transpose of a rows x cols row-major matrix, where cols % rows == 0 or rows % cols == 0
    void transpose_inplace(C* a, size_t rows, size_t cols, const parallel_for_t* par) const
    {
        if (rows == 1 || cols == 1)
            return; // a vector: memory layout is already the transpose
        if (cols % rows == 0)
        {
            // (i,k,j) -> (k,i,j): move rows of length `rows` around, then transpose each rows x rows block
            chunk_transpose(a, rows, cols / rows, rows);
            square_transposes(a, rows, cols / rows, par);
        }
        else
        {
            square_transposes(a, cols, rows / cols, par);
            chunk_transpose(a, rows / cols, cols, cols);
        }
    }

    // a is a (g_rows x g_cols) row-major grid of chunks of `len` complex numbers; transpose the grid
    // (cycle following, one chunk of scratch). Serial.
    void chunk_transpose(C* a, size_t g_rows, size_t g_cols, size_t len) const
    {
        const size_t total = g_rows * g_cols;
        if (total <= 2 || g_rows == 1 || g_cols == 1)
            return; // the grid is a single row or column: same memory layout
        std::vector<bool> seen(total, false);
        std::vector<C> tmp(len);
        auto chunk = [&](size_t p) { return a + p * len; };
        // destination q = b*g_rows + a_  <-  source p = a_*g_cols + b ; inverse: q -> p
        auto src_of = [&](size_t q) { return (q % g_rows) * g_cols + q / g_rows; };
        for (size_t s = 1; s + 1 < total; ++s)
        {
            if (seen[s])
                continue;
            std::memcpy(static_cast<void*>(tmp.data()), chunk(s), len * sizeof(C));
            size_t cur = s;
            for (;;)
            {
                seen[cur]  = true;
                size_t src = src_of(cur);
                if (src == s)
                {
                    std::memcpy(static_cast<void*>(chunk(cur)), tmp.data(), len * sizeof(C));
                    break;
                }
                std::memcpy(static_cast<void*>(chunk(cur)), chunk(src), len * sizeof(C));
                cur = src;
            }
        }
    }

    // transpose `count` consecutive n x n blocks in place; parallel over (block, tile row)
    static constexpr size_t TS = 8;
    void square_transposes(C* a, size_t n, size_t count, const parallel_for_t* par) const
    {
        const size_t nt = n / TS;
        for_each(count * nt, par, [&](size_t item, size_t) { square_tile_row(a + (item / nt) * n * n, n, item % nt); });
        for (size_t k = 0; k < count; ++k) // pairs (i,j), i<j, not covered by full tiles
        {
            C* m = a + k * n * n;
            for (size_t i = 0; i < n; ++i)
                for (size_t j = std::max(i + 1, nt * TS); j < n; ++j) std::swap(m[i * n + j], m[j * n + i]);
        }
    }
    // tiles (I,I) and (I,J), J > I of an n x n matrix: 8x8 tiles are loaded into local arrays (so the
    // compiler can keep them in registers and use shuffles) and written back transposed
    static void square_tile_row(C* a, size_t n, size_t I)
    {
        const size_t nt = n / TS;
        C ta[TS][TS], tb[TS][TS];
        C* d = a + I * TS * n + I * TS;
        for (size_t i = 0; i < TS; ++i)
            for (size_t j = 0; j < TS; ++j) ta[i][j] = d[i * n + j];
        for (size_t i = 0; i < TS; ++i)
            for (size_t j = 0; j < TS; ++j) d[i * n + j] = ta[j][i];
        for (size_t J = I + 1; J < nt; ++J)
        {
            C* p = a + I * TS * n + J * TS; // tile (I,J)
            C* q = a + J * TS * n + I * TS; // tile (J,I)
            if (J + 1 < nt)
            {
                for (size_t i = 0; i < TS; ++i) __builtin_prefetch(q + TS * n + i * n, 1);
            }
            for (size_t i = 0; i < TS; ++i)
                for (size_t j = 0; j < TS; ++j)
                {
                    ta[i][j] = p[i * n + j];
                    tb[i][j] = q[i * n + j];
                }
            for (size_t i = 0; i < TS; ++i)
                for (size_t j = 0; j < TS; ++j)
                {
                    p[i * n + j] = tb[j][i];
                    q[i * n + j] = ta[j][i];
                }
        }
    }

    workspace& own_ws()
    {
        if (own.buf.size() == 0)
            own = make_workspace();
        return own;
    }

    void cexp_m(size_t m, T* d) const
    {
        const long double a = -2.0L * 3.141592653589793238462643383279502884L * (long double)m / (long double)N;
        d[0] = T(std::cos(a));
        d[1] = T(std::sin(a));
    }
    // KFR handles the length without Bluestein (no prime factor left > 10 after radices 2..10)
    static bool kfr_fast(size_t n) { return fs_kfr_fast(n); }
    // time per point of a length-`len` KFR DFT, in cache; `effort` scales the measurement time
    static double cost_per_point(size_t len, int effort)
    {
        fs_sub_dft<T> p(len);
        kfr::univector<C> a(len, C(T(1), T(0.5)));
        kfr::univector<kfr::u8> tmp(p.temp_size);
        p.execute(a.data(), a.data(), tmp.data(), kfr::cfalse);
        const size_t reps = std::max<size_t>(4, (size_t(effort) << 19) / len);
        double best       = 1e30;
        for (int round = 0; round < 2 * effort; ++round)
        {
            const double t0 = now();
            for (size_t r = 0; r < reps; ++r) p.execute(a.data(), a.data(), tmp.data(), kfr::cfalse);
            best = std::min(best, (now() - t0) / double(reps * len));
        }
        return best;
    }
    static size_t default_max_bytes(const options& opt)
    {
        if (opt.max_bytes)
            return opt.max_bytes;
        const size_t l2 = l2_cache_bytes();
        return l2 ? std::min<size_t>(std::max<size_t>(l2 / 2, size_t(128) << 10), size_t(1) << 20) : size_t(1) << 20;
    }
    // all N1 with a usable N1 * N2 split (both factors fast in KFR if possible, within the length limits)
    static std::vector<size_t> all_splits(size_t n, const options& opt)
    {
        size_t maxlen = std::max<size_t>(default_max_bytes(opt) / sizeof(C), opt.min_len);
        std::vector<size_t> cand;
        for (int attempt = 0; attempt < 8 && cand.empty(); ++attempt, maxlen *= 4)
            for (size_t d = opt.min_len; d <= maxlen && d * opt.min_len <= n; ++d)
            {
                if (n % d != 0)
                    continue;
                const size_t e = n / d;
                if (e >= opt.min_len && e <= maxlen && kfr_fast(d) && kfr_fast(e))
                    cand.push_back(d);
            }
        // No split with two fast factors (n has a prime factor >= 101): allow one factor that needs Bluestein
        // (fs_sub_dft runs it with a smooth convolution length), keeping the other one fast.
        maxlen = std::max<size_t>(default_max_bytes(opt) / sizeof(C), opt.min_len);
        for (int attempt = 0; attempt < 3 && cand.empty(); ++attempt, maxlen *= 2)
            for (size_t d = opt.min_len; d <= maxlen && d * opt.min_len <= n; ++d)
            {
                if (n % d != 0)
                    continue;
                const size_t e = n / d;
                if (e >= opt.min_len && e <= maxlen && (kfr_fast(d) || kfr_fast(e)))
                    cand.push_back(d);
            }
        return cand;
    }
    static bool divisible_split(size_t n, size_t d) { return (n / d) % d == 0 || d % (n / d) == 0; }

    static size_t choose_n1(size_t n, const options& opt)
    {
        std::vector<size_t> cand = all_splits(n, opt);
        if (cand.empty())
            return 0;
        std::vector<size_t> cdiv; // splits that allow in-place execution without a work buffer
        for (size_t d : cand)
            if (divisible_split(n, d))
                cdiv.push_back(d);
        const double s = std::sqrt(double(n));
        std::map<std::pair<size_t, int>, double> cost;
        auto c = [&](size_t len, int effort)
        {
            auto it = cost.find({ len, effort });
            return it != cost.end() ? it->second : (cost[{ len, effort }] = cost_per_point(len, effort));
        };
        auto total = [&](size_t d, int effort)
        { return opt.measure ? c(d, effort) + c(n / d, effort) : std::fabs(std::log(d / s)); };
        // rank by a quick measurement, then re-measure the best few with more effort: single quick
        // timings are noisy (a few percent) and the splits at the top of the list are close to each other
        auto shortlist = [&](std::vector<size_t> v)
        {
            std::sort(v.begin(), v.end(), [&](size_t a, size_t b) { return total(a, 1) < total(b, 1); });
            if (opt.measure && v.size() > 3)
                v.resize(3);
            return v;
        };
        auto best_of = [&](const std::vector<size_t>& v)
        {
            const int effort = opt.measure ? 4 : 1;
            return *std::min_element(v.begin(), v.end(), [&](size_t a, size_t b) { return total(a, effort) < total(b, effort); });
        };
        const size_t any = best_of(shortlist(cand));
        if (!opt.prefer_inplace || cdiv.empty())
            return any;
        const size_t div = best_of(shortlist(cdiv));
        if (!opt.measure)
            return div;
        return total(div, 4) <= opt.inplace_slack * total(any, 4) ? div : any;
    }

    size_t N, N1 = 1, N2 = 1, B, pf, Lbits = 0;
    int oop_strategy = -1;
    fs_sub_dft<T> p1, p2;
    std::vector<T> lo, hi, wj;
    workspace own;
};
