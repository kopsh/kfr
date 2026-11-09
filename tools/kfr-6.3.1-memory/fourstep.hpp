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
        size_t max_bytes  = size_t(1) << 20; // max sub-FFT size in bytes (keep it near L2)
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
        p1 = kfr::dft_plan<T>(N1);
        p2 = kfr::dft_plan<T>(N2);
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
    static size_t find_split(size_t n, bool inplace = false)
    {
        options o;
        o.measure        = false;
        o.prefer_inplace = inplace;
        return choose_n1(n, o);
    }

    /// Per-call scratch memory (a few MB). One per thread for concurrent calls.
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
        w.buf.resize(B * std::max(N1, N2));
        return w;
    }

    size_t size() const { return N; }
    size_t n1() const { return N1; }
    size_t n2() const { return N2; }
    /// True if execute_inplace() needs no N-point buffer for this plan.
    bool inplace_without_work() const { return N2 % N1 == 0 || N1 % N2 == 0; }
    /// Memory owned by the plan (twiddles, sub-plans) plus one workspace.
    size_t extra_bytes() const
    {
        return p1.data_size + p2.data_size + (lo.size() + hi.size() + wj.size()) * sizeof(T) +
               std::max(p1.temp_size, p2.temp_size) + B * std::max(N1, N2) * sizeof(C);
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
                     workspace& ws   = wss[w];
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
            {
                if (n % d != 0)
                    continue;
                const size_t e = n / d;
                if (e >= opt.min_len && e <= maxlen && kfr_fast(d) && kfr_fast(e))
                    cand.push_back(d);
            }
        if (cand.empty())
            return 0;
        std::vector<size_t> cdiv; // splits that allow in-place execution without a work buffer
        for (size_t d : cand)
            if ((n / d) % d == 0 || d % (n / d) == 0)
                cdiv.push_back(d);
        const double s = std::sqrt(double(n));
        std::map<size_t, double> cost;
        auto c = [&](size_t len)
        {
            auto it = cost.find(len);
            return it != cost.end() ? it->second : (cost[len] = cost_per_point(len));
        };
        auto total = [&](size_t d)
        { return opt.measure ? c(d) + c(n / d) : std::fabs(std::log(d / s)); };
        auto best_of = [&](const std::vector<size_t>& v)
        { return *std::min_element(v.begin(), v.end(), [&](size_t a, size_t b) { return total(a) < total(b); }); };
        const size_t any = best_of(cand);
        if (!opt.prefer_inplace || cdiv.empty())
            return any;
        const size_t div = best_of(cdiv);
        if (!opt.measure)
            return div;
        return total(div) <= opt.inplace_slack * total(any) ? div : any;
    }

    size_t N, N1 = 1, N2 = 1, B, pf, Lbits = 0;
    int oop_strategy = -1;
    kfr::dft_plan<T> p1, p2;
    std::vector<T> lo, hi, wj;
    workspace own;
};
