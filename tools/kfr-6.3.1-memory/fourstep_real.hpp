// Real-input four-step FFT (N real samples <-> N/2 + 1 complex bins), built on fft_fourstep<T>.
//
// The N real samples are read as M = N/2 complex numbers z[n] = x[2n] + i*x[2n+1] (no copy), transformed with
// the complex four-step FFT of size M, and a single in-place pass over the pairs (k, M-k) turns that into the
// spectrum of the real signal. Compared with a complex transform of the same data this halves memory and time.
//
// Layout (same as kfr::dft_plan_real with dft_pack_format::CCs): the spectrum has M + 1 complex bins, the
// imaginary parts of bin 0 and bin M are zero. The inverse is unnormalized (returns N * x).
//
//   execute_inplace(x)       x: N + 2 reals, first N hold the signal; result: M + 1 complex.   memory: x only
//   execute(out, in)         in (N reals) is preserved, out has M + 1 complex.                 memory: in + out
//   inverse_inplace(x)       x: M + 1 complex; result: N reals at the start of x.             memory: x only
//   inverse(out, in)         in (M + 1 complex) is preserved, out has N reals.                 memory: in + out
//
// Needs fft_fourstep<T>::find_split(N / 2) != 0. Thread-safety and opt-in multi-threading work as in fft_fourstep.
#pragma once
#include "fourstep.hpp"

template <typename T>
class fft_fourstep_real
{
public:
    using C              = kfr::complex<T>;
    using core_t         = fft_fourstep<T>;
    using workspace      = typename core_t::workspace;
    using options        = typename core_t::options;
    using parallel_for_t = typename core_t::parallel_for_t;

    explicit fft_fourstep_real(size_t n) : fft_fourstep_real(n, core_t::for_inplace()) {}
    fft_fourstep_real(size_t n, const options& opt) : N(n), M(n / 2), core(n / 2, opt)
    {
        if (n % 2 != 0 || n < 4)
            throw std::invalid_argument("fft_fourstep_real: n must be even");
        while ((size_t(1) << (2 * Lbits)) < M) ++Lbits;
        const size_t L = size_t(1) << Lbits;
        lo.resize(2 * L);
        hi.resize(2 * (M / 2 / L + 2));
        for (size_t r = 0; r < L; ++r) cexp_n(r, lo.data() + 2 * r);
        for (size_t q = 0; q < hi.size() / 2; ++q) cexp_n(q * L, hi.data() + 2 * q);
    }

    static size_t find_split(size_t n) { return n % 2 ? 0 : core_t::find_split(n / 2, true); }
    /// See fft_fourstep::autotune; tunes the complex transform of size n / 2 that this class runs.
    static typename core_t::tune_result autotune(size_t n, const options& base = options{}, bool inplace = true,
                                                  int reps = 2, size_t max_candidates = 6)
    {
        return core_t::autotune(n / 2, base, inplace, reps, max_candidates);
    }

    size_t size() const { return N; }
    size_t complex_bins() const { return M + 1; }
    const core_t& core_plan() const { return core; }
    workspace make_workspace() const { return core.make_workspace(); }
    std::vector<workspace> make_workspaces(size_t count) const { return core.make_workspaces(count); }

    // ---- forward
    void execute_inplace(T* x, workspace& ws) const { forward(x, &ws, nullptr); }
    void execute(C* out, const T* in, workspace& ws) const
    {
        std::memcpy(static_cast<void*>(out), in, N * sizeof(T));
        forward(reinterpret_cast<T*>(out), &ws, nullptr);
    }
    // ---- inverse (unnormalized)
    void inverse_inplace(T* x, workspace& ws) const { inverse_pass(x, x, &ws, nullptr); }
    void inverse(T* out, const C* in, workspace& ws) const { inverse_pass(reinterpret_cast<const T*>(in), out, &ws, nullptr); }

    // ---- opt-in multi-threading (see fft_fourstep::parallel_for_t)
    void execute_inplace(T* x, std::vector<workspace>& wss, const parallel_for_t& par) const { forward(x, wss.data(), &par); }
    void execute(C* out, const T* in, std::vector<workspace>& wss, const parallel_for_t& par) const
    {
        copy_reals(reinterpret_cast<T*>(out), in, &par);
        forward(reinterpret_cast<T*>(out), wss.data(), &par);
    }
    void inverse_inplace(T* x, std::vector<workspace>& wss, const parallel_for_t& par) const
    {
        inverse_pass(x, x, wss.data(), &par);
    }
    void inverse(T* out, const C* in, std::vector<workspace>& wss, const parallel_for_t& par) const
    {
        inverse_pass(reinterpret_cast<const T*>(in), out, wss.data(), &par);
    }

private:
    static constexpr size_t PAIRS_PER_ITEM = size_t(1) << 15;

    void cexp_n(size_t m, T* d) const
    {
        const long double a = -2.0L * 3.141592653589793238462643383279502884L * (long double)m / (long double)N;
        d[0] = T(std::cos(a));
        d[1] = T(std::sin(a));
    }
    // w_N^k = exp(-2*pi*i*k/N), k < N/2
    void tw(size_t k, T& wr, T& wi) const
    {
        const size_t mask = (size_t(1) << Lbits) - 1;
        const T* a        = lo.data() + 2 * (k & mask);
        const T* b        = hi.data() + 2 * (k >> Lbits);
        wr                = a[0] * b[0] - a[1] * b[1];
        wi                = a[0] * b[1] + a[1] * b[0];
    }

    template <class F>
    static void for_each(size_t count, const parallel_for_t* par, F&& f)
    {
        if (par && *par && count > 1)
            (*par)(count, std::function<void(size_t, size_t)>(std::ref(f)));
        else
            for (size_t i = 0; i < count; ++i) f(i, size_t(0));
    }
    void copy_reals(T* out, const T* in, const parallel_for_t* par) const
    {
        const size_t parts = par ? 64 : 1, per = (N + parts - 1) / parts;
        for_each(parts, par,
                 [&](size_t i, size_t)
                 {
                     const size_t a = i * per, e = std::min(N, a + per);
                     if (a < e)
                         std::memcpy(out + a, in + a, (e - a) * sizeof(T));
                 });
    }
    // pairs k = 1 .. (M-1)/2 (k < M-k); for even M the middle bin k = M/2 pairs with itself
    size_t pair_count() const { return (M - 1) / 2; }
    size_t pair_items() const { return (pair_count() + PAIRS_PER_ITEM - 1) / PAIRS_PER_ITEM; }

    // z: M + 1 complex; the first M hold the signal as complex numbers
    void forward(T* z, workspace* wss, const parallel_for_t* par) const
    {
        core.execute_inplace(reinterpret_cast<C*>(z), false, wss, par); // Z = DFT_M(z)
        const T z0r = z[0], z0i = z[1];
        for_each(pair_items(), par,
                 [&](size_t item, size_t)
                 {
                     const size_t k0 = 1 + item * PAIRS_PER_ITEM, k1 = std::min(k0 + PAIRS_PER_ITEM, pair_count() + 1);
                     for (size_t k = k0; k < k1; ++k)
                     {
                         // E = (Z[k] + conj Z[M-k]) / 2, O = -i (Z[k] - conj Z[M-k]) / 2
                         // X[k] = E + w^k O, X[M-k] = conj(E - w^k O)
                         const size_t j = M - k;
                         const T ar = z[2 * k], ai = z[2 * k + 1], br = z[2 * j], bi = -z[2 * j + 1];
                         const T Er = (ar + br) * T(0.5), Ei = (ai + bi) * T(0.5);
                         const T Dr = (ar - br) * T(0.5), Di = (ai - bi) * T(0.5);
                         const T Or = Di, Oi = -Dr;
                         T wr, wi;
                         tw(k, wr, wi);
                         const T tr = wr * Or - wi * Oi, ti = wr * Oi + wi * Or;
                         z[2 * k]     = Er + tr;
                         z[2 * k + 1] = Ei + ti;
                         z[2 * j]     = Er - tr;
                         z[2 * j + 1] = ti - Ei;
                     }
                 });
        if (M % 2 == 0) // middle bin pairs with itself: X = conj(Z)
            z[2 * (M / 2) + 1] = -z[2 * (M / 2) + 1];
        z[0]         = z0r + z0i;
        z[1]         = 0;
        z[2 * M]     = z0r - z0i;
        z[2 * M + 1] = 0;
    }

    // src: M + 1 complex bins, dst: room for M complex (N reals); src == dst is allowed
    void inverse_pass(const T* src, T* dst, workspace* wss, const parallel_for_t* par) const
    {
        const T x0r = src[0], xmr = src[2 * M];
        for_each(pair_items(), par,
                 [&](size_t item, size_t)
                 {
                     const size_t k0 = 1 + item * PAIRS_PER_ITEM, k1 = std::min(k0 + PAIRS_PER_ITEM, pair_count() + 1);
                     for (size_t k = k0; k < k1; ++k)
                     {
                         // Z = 2 (E + i O) with E2 = X[k] + conj X[M-k], O2 = conj(w^k) (X[k] - conj X[M-k])
                         const size_t j = M - k;
                         const T ar = src[2 * k], ai = src[2 * k + 1], cr = src[2 * j], ci = -src[2 * j + 1];
                         const T E2r = ar + cr, E2i = ai + ci, D2r = ar - cr, D2i = ai - ci;
                         T wr, wi;
                         tw(k, wr, wi);
                         const T O2r = wr * D2r + wi * D2i, O2i = wr * D2i - wi * D2r;
                         dst[2 * k]     = E2r - O2i;
                         dst[2 * k + 1] = E2i + O2r;
                         dst[2 * j]     = E2r + O2i;
                         dst[2 * j + 1] = O2r - E2i;
                     }
                 });
        if (M % 2 == 0)
        {
            const size_t k = M / 2;
            const T re = src[2 * k], im = src[2 * k + 1];
            dst[2 * k]     = 2 * re;
            dst[2 * k + 1] = -2 * im;
        }
        dst[0] = x0r + xmr;
        dst[1] = x0r - xmr;
        core.execute_inplace(reinterpret_cast<C*>(dst), true, wss, par);
    }

    size_t N, M, Lbits = 0;
    core_t core;
    std::vector<T> lo, hi;
};
