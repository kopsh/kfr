// Four-step FFT on top of KFR's public dft_plan: N = N1 * N2.
// Extra memory is O(sqrt(N)) instead of KFR's ~2N complex (twiddles + scratch).
#pragma once
#include <kfr/dft.hpp>
#include <cmath>
#include <vector>

template <typename T>
class fft_fourstep
{
public:
    using C = kfr::complex<T>;
    explicit fft_fourstep(size_t n, size_t block = 32) : N(n), B(block)
    {
        N1 = pick_n1(n);
        N2 = n / N1;
        p1 = kfr::dft_plan<T>(N1);
        p2 = kfr::dft_plan<T>(N2);
        temp.resize(std::max(p1.temp_size, p2.temp_size));
        buf.resize(B * std::max(N1, N2));
        Lbits = 0;
        while ((size_t(1) << (2 * Lbits)) < N) ++Lbits; // L ~ sqrt(N), power of two
        const size_t L = size_t(1) << Lbits;
        lo.resize(L);
        hi.resize(N / L + 1);
        for (size_t r = 0; r < L; ++r) lo[r] = cexp_m(r);
        for (size_t q = 0; q < hi.size(); ++q) hi[q] = cexp_m(q * L);
    }
    size_t n1() const { return N1; }
    size_t n2() const { return N2; }
    size_t extra_bytes() const
    {
        return p1.data_size + p2.data_size + temp.size() + buf.size() * sizeof(C) +
               (lo.size() + hi.size()) * sizeof(C);
    }

    // out-of-place; `in` is used as work buffer and destroyed. out != in.
    void execute(C* out, C* in, bool inverse) const
    {
        C* b = const_cast<C*>(buf.data());
        kfr::u8* t = const_cast<kfr::u8*>(temp.data());
        // step 1: length-N1 DFTs over columns n2, then twiddle w_N^(n2*k1)
        for (size_t c0 = 0; c0 < N2; c0 += B)
        {
            const size_t nb = std::min(B, N2 - c0);
            for (size_t n1 = 0; n1 < N1; ++n1)
            {
                const C* row = in + n1 * N2 + c0;
                for (size_t j = 0; j < nb; ++j) b[j * N1 + n1] = row[j];
            }
            for (size_t j = 0; j < nb; ++j)
            {
                C* col = b + j * N1;
                p1.execute(col, col, t, inverse);
                const size_t n2 = c0 + j;
                size_t m = 0;
                for (size_t k1 = 0; k1 < N1; ++k1, m += n2)
                {
                    C w = lo[m & ((size_t(1) << Lbits) - 1)] * hi[m >> Lbits];
                    col[k1] *= inverse ? std::conj(w) : w;
                }
            }
            for (size_t k1 = 0; k1 < N1; ++k1)
            {
                C* row = in + k1 * N2 + c0;
                for (size_t j = 0; j < nb; ++j) row[j] = b[j * N1 + k1];
            }
        }
        // step 2: length-N2 DFTs over rows k1, write transposed: out[k1 + N1*k2]
        for (size_t r0 = 0; r0 < N1; r0 += B)
        {
            const size_t nb = std::min(B, N1 - r0);
            for (size_t j = 0; j < nb; ++j)
                p2.execute(b + j * N2, in + (r0 + j) * N2, t, inverse);
            for (size_t k2 = 0; k2 < N2; ++k2)
            {
                C* o = out + k2 * N1 + r0;
                for (size_t j = 0; j < nb; ++j) o[j] = b[j * N2 + k2];
            }
        }
    }

private:
    C cexp_m(size_t m) const
    {
        const long double a = -2.0L * 3.141592653589793238462643383279502884L * (long double)m / (long double)N;
        return C(T(std::cos(a)), T(std::sin(a)));
    }
    static bool kfr_fast(size_t n) // no prime factor > 10 left -> no Bluestein inside KFR
    {
        for (size_t r = 10; r >= 2; --r)
            while (n % r == 0) n /= r;
        return n < 101;
    }
    static size_t pick_n1(size_t n)
    {
        size_t best = 1;
        const double s = std::sqrt(double(n));
        for (size_t d = 1; d * d <= n; ++d)
            if (n % d == 0 && kfr_fast(d) && kfr_fast(n / d) &&
                std::fabs(std::log(double(d) / s)) < std::fabs(std::log(double(best) / s)))
                best = d;
        return best;
    }
    size_t N, N1 = 1, N2 = 1, B, Lbits = 0;
    kfr::dft_plan<T> p1, p2;
    kfr::univector<kfr::u8> temp;
    kfr::univector<C> buf;
    std::vector<C> lo, hi;
};
