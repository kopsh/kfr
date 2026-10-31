/*
 * Built with Clang together with KFR. Exceptions never cross the C interface.
 */
#include "fft_wrapper.h"

#include <kfr/dft.hpp>
#include <kfr/version.hpp>
#include <new>

namespace kfr
{
const char* library_version_dft();
}

struct fft_wrapper_plan
{
    explicit fft_wrapper_plan(size_t size) : plan(size), temp(plan.temp_size) {}
    kfr::dft_plan<float> plan;
    kfr::univector<kfr::u8> temp;
};

extern "C" fft_wrapper_plan* fft_wrapper_create_f32(size_t size)
{
    try
    {
        return new fft_wrapper_plan(size);
    }
    catch (...)
    {
        return nullptr;
    }
}

static int execute(fft_wrapper_plan* p, float* out, const float* in, bool inverse)
{
    if (!p || !out || !in)
        return -1;
    try
    {
        p->plan.execute(reinterpret_cast<kfr::complex<float>*>(out),
                        reinterpret_cast<const kfr::complex<float>*>(in), p->temp.data(), inverse);
        return 0;
    }
    catch (...)
    {
        return -1;
    }
}

extern "C" int fft_wrapper_forward_f32(fft_wrapper_plan* p, float* out, const float* in)
{
    return execute(p, out, in, false);
}

extern "C" int fft_wrapper_inverse_f32(fft_wrapper_plan* p, float* out, const float* in)
{
    return execute(p, out, in, true);
}

extern "C" void fft_wrapper_destroy(fft_wrapper_plan* p) { delete p; }

extern "C" const char* fft_wrapper_version(void) { return kfr::library_version_dft(); }
