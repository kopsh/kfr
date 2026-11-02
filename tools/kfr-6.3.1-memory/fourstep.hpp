// Four-step FFT for very large sizes, built on KFR 6.3.1's public dft_plan.
//
//   N = N1 * N2,  x[n1*N2 + n2]  ->  X[k1 + N1*k2]
//   step 1: for every column n2: length-N1 DFT, multiply by w_N^(n2*k1)
//   step 2: for every row k1:    length-N2 DFT, store transposed
//
// Extra memory is a few MB (two small KFR plans, sqrt(N)-sized twiddle tables, a block buffer)
// instead of ~2N complex for dft_plan<T>(N) (twiddles + scratch). Every element is read and
// written twice from main memory, compared with once per radix stage for dft_plan.
//
// Results match dft_plan<T>::execute: natural order, inverse is unnormalized.
#pragma once
#include <kfr/dft.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#ifndef FS_PROFILE
#define FS_PROFILE 0
#endif

template <typename T>
class fft_fourstep
{
public:
    using C = kfr::complex<T>;

    struct options
    {
        size_t n1         = 0;  // force N1 (0 = choose automatically)
        bool measure      = true; // time candidate sub-FFT lengths on this machine
        size_t min_len    = 512;
        size_t max_bytes  = size_t(1) << 20; // max sub-FFT size in bytes (keep it near L2)
        size_t block      = 32; // columns/rows per block
        size_t prefetch   = 8;  // prefetch distance in rows (0 = off)
    };

    explicit fft_fourstep(size_t n) : fft_fourstep(n, options{}) {}
    fft_fourstep(size_t n, const options& opt) : N(n), B(opt.block), pf(opt.prefetch)
    {
        N1 = opt.n1 ? opt.n1 : choose_n1(n, opt);
        if (N1 == 0 || n % N1 != 0)
            throw std::invalid_argument("fft_fourstep: no usable N1 * N2 split");
        N2 = n / N1;
        p1 = kfr::dft_plan<T>(N1);
        p2 = kfr::dft_plan<T>(N2);
        temp.resize(std::max(p1.temp_size, p2.temp_size));
        buf.resize(B * std::max(N1, N2));
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

    /// Returns a usable split for n, or 0 when there is none (fall back to dft_plan<T>(n) then).
    static size_t find_split(size_t n) { return choose_n1(n, options{ 0, false }); }

    size_t size() const { return N; }
    size_t n1() const { return N1; }
    size_t n2() const { return N2; }
    /// Memory owned by this object (excluding an internal work buffer, see work_bytes()).
    size_t extra_bytes() const
    {
        return p1.data_size + p2.data_size + temp.size() + buf.size() * sizeof(C) +
               (lo.size() + hi.size() + wj.size()) * sizeof(T);
    }
    size_t work_bytes() const { return work ? N * sizeof(C) : 0; }
    void release_work() { work.reset(); }

    /// Out-of-place, `in` is overwritten (used as work area). Extra memory: none.
    void execute_destroy(C* out, C* in, bool inverse = false) { dispatch(out, in, in, inverse); }

    /// Out-of-place, `in` preserved. Uses `work` (N complex) or an internal work buffer.
    void execute(C* out, const C* in, bool inverse = false, C* work_area = nullptr)
    {
        dispatch(out, in, work_area ? work_area : get_work(), inverse);
    }

    /// In-place. Uses `work` (N complex) or an internal work buffer.
    void execute_inplace(C* inout, bool inverse = false, C* work_area = nullptr)
    {
        dispatch(inout, inout, work_area ? work_area : get_work(), inverse);
    }

#if FS_PROFILE
    double prof[5] = {}; // gather, fft1, twiddle+scatter, fft2, transpose
#endif

private:
    C* get_work()
    {
        if (!work)
            work.reset(new C[N]);
        return work.get();
    }
    static double now()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void dispatch(C* dst, const C* src, C* wrk, bool inverse)
    {
        if (inverse)
            run<true>(dst, src, wrk);
        else
            run<false>(dst, src, wrk);
    }

    // src -> wrk (step 1, row layout), wrk -> dst (step 2, transposed). wrk may equal src; dst != wrk.
    template <bool inv>
    void run(C* dst, const C* src, C* wrk)
    {
        const T* x        = reinterpret_cast<const T*>(src);
        T* w              = reinterpret_cast<T*>(wrk);
        T* o              = reinterpret_cast<T*>(dst);
        T* b              = reinterpret_cast<T*>(buf.data());
        kfr::u8* t        = temp.data();
        const size_t mask = (size_t(1) << Lbits) - 1;
        const size_t lb   = 2 * sizeof(T); // bytes per complex
#if FS_PROFILE
        double t0 = now(), t1;
#define FS_TICK(i) (t1 = now(), prof[i] += t1 - t0, t0 = t1)
#else
#define FS_TICK(i) ((void)0)
#endif
        for (size_t c0 = 0; c0 < N2; c0 += B)
        {
            const size_t nb = std::min(B, N2 - c0);
            for (size_t n1 = 0; n1 < N1; ++n1)
            {
                const T* row = x + 2 * (n1 * N2 + c0);
                if (pf && n1 + pf < N1)
                {
                    const char* p = reinterpret_cast<const char*>(x + 2 * ((n1 + pf) * N2 + c0));
                    for (size_t q = 0; q < nb * lb; q += 64) __builtin_prefetch(p + q);
                }
                T* d = b + 2 * n1;
                for (size_t j = 0; j < nb; ++j)
                {
                    d[2 * j * N1]     = row[2 * j];
                    d[2 * j * N1 + 1] = row[2 * j + 1];
                }
            }
            FS_TICK(0);
            for (size_t j = 0; j < nb; ++j)
            {
                C* col = buf.data() + j * N1;
                p1.execute(col, col, t, kfr::cbool<inv>);
            }
            FS_TICK(1);
            size_t m = 0; // c0 * k1
            for (size_t k1 = 0; k1 < N1; ++k1, m += c0)
            {
                const T* L0 = lo.data() + 2 * (m & mask);
                const T* H0 = hi.data() + 2 * (m >> Lbits);
                const T sr  = L0[0] * H0[0] - L0[1] * H0[1];
                const T si  = L0[0] * H0[1] + L0[1] * H0[0];
                const T* tw = wj.data() + 2 * k1 * B;
                T* row      = w + 2 * (k1 * N2 + c0);
                if (pf && k1 + pf < N1)
                {
                    char* p = reinterpret_cast<char*>(w + 2 * ((k1 + pf) * N2 + c0));
                    for (size_t q = 0; q < nb * lb; q += 64) __builtin_prefetch(p + q, 1);
                }
                const T* s = b + 2 * k1;
                for (size_t j = 0; j < nb; ++j)
                {
                    const T wr     = sr * tw[2 * j] - si * tw[2 * j + 1];
                    const T wi     = inv ? -(sr * tw[2 * j + 1] + si * tw[2 * j])
                                         : sr * tw[2 * j + 1] + si * tw[2 * j];
                    const T dr     = s[2 * j * N1];
                    const T di     = s[2 * j * N1 + 1];
                    row[2 * j]     = dr * wr - di * wi;
                    row[2 * j + 1] = dr * wi + di * wr;
                }
            }
            FS_TICK(2);
        }
        for (size_t r0 = 0; r0 < N1; r0 += B)
        {
            const size_t nb = std::min(B, N1 - r0);
            for (size_t j = 0; j < nb; ++j)
                p2.execute(buf.data() + j * N2, wrk + (r0 + j) * N2, t, kfr::cbool<inv>);
            FS_TICK(3);
            for (size_t k2 = 0; k2 < N2; ++k2)
            {
                T* d = o + 2 * (k2 * N1 + r0);
                if (pf && k2 + pf < N2)
                {
                    char* p = reinterpret_cast<char*>(o + 2 * ((k2 + pf) * N1 + r0));
                    for (size_t q = 0; q < nb * lb; q += 64) __builtin_prefetch(p + q, 1);
                }
                const T* s = b + 2 * k2;
                for (size_t j = 0; j < nb; ++j)
                {
                    d[2 * j]     = s[2 * j * N2];
                    d[2 * j + 1] = s[2 * j * N2 + 1];
                }
            }
            FS_TICK(4);
        }
#undef FS_TICK
    }

    void cexp_m(size_t m, T* d) const
    {
        const long double a = -2.0L * 3.141592653589793238462643383279502884L * (long double)m / (long double)N;
        d[0] = T(std::cos(a));
        d[1] = T(std::sin(a));
    }
    // KFR handles the length without Bluestein (no prime factor left > 10 after radices 2..10)
    static bool kfr_fast(size_t n)
    {
        for (size_t r = 10; r >= 2; --r)
            while (n % r == 0) n /= r;
        return n < 101;
    }
    static double cost_per_point(size_t len)
    {
        kfr::dft_plan<T> p(len);
        kfr::univector<C> a(len, C(T(1), T(0.5)));
        kfr::univector<kfr::u8> tmp(p.temp_size);
        p.execute(a.data(), a.data(), tmp.data());
        const size_t reps = std::max<size_t>(4, (size_t(1) << 19) / len);
        double best       = 1e30;
        for (int round = 0; round < 2; ++round)
        {
            const double t0 = now();
            for (size_t r = 0; r < reps; ++r) p.execute(a.data(), a.data(), tmp.data());
            best = std::min(best, (now() - t0) / double(reps * len));
        }
        return best;
    }
    static size_t choose_n1(size_t n, const options& opt)
    {
        size_t maxlen = std::max<size_t>(opt.max_bytes / sizeof(C), opt.min_len);
        std::vector<size_t> cand;
        for (int attempt = 0; attempt < 8 && cand.empty(); ++attempt, maxlen *= 4)
            for (size_t d = opt.min_len; d <= maxlen && d * opt.min_len <= n; ++d)
                if (n % d == 0 && n / d >= opt.min_len && n / d <= maxlen && kfr_fast(d) && kfr_fast(n / d))
                    cand.push_back(d);
        if (cand.empty())
            return 0;
        if (!opt.measure)
        {
            const double s = std::sqrt(double(n));
            return *std::min_element(cand.begin(), cand.end(), [&](size_t a, size_t b)
                                     { return std::fabs(std::log(a / s)) < std::fabs(std::log(b / s)); });
        }
        std::map<size_t, double> cost;
        auto c = [&](size_t len)
        {
            auto it = cost.find(len);
            return it != cost.end() ? it->second : (cost[len] = cost_per_point(len));
        };
        size_t best = cand[0];
        for (size_t d : cand)
            if (c(d) + c(n / d) < c(best) + c(n / best))
                best = d;
        return best;
    }

    size_t N, N1 = 1, N2 = 1, B, pf, Lbits = 0;
    kfr::dft_plan<T> p1, p2;
    kfr::univector<kfr::u8> temp;
    kfr::univector<C> buf;
    std::vector<T> lo, hi, wj;
    std::unique_ptr<C[]> work;
};
