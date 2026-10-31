/*
 * Example public interface of an FFT library built with Clang on top of KFR.
 * Only C types cross the boundary, so callers may use any compiler (here GCC 10).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct fft_wrapper_plan fft_wrapper_plan;

/* Creates a complex single-precision DFT plan of the given size. Returns NULL on failure. */
fft_wrapper_plan* fft_wrapper_create_f32(size_t size);

/* Interleaved complex input/output (re, im, re, im, ...), 2 * size floats each.
   Returns 0 on success, -1 on failure. */
int fft_wrapper_forward_f32(fft_wrapper_plan* plan, float* out, const float* in);
int fft_wrapper_inverse_f32(fft_wrapper_plan* plan, float* out, const float* in);

void fft_wrapper_destroy(fft_wrapper_plan* plan);

/* KFR version string, e.g. "KFR 6.3.1 optimized neon64 64-bit (clang-15.0.0/linux) +in +ve" */
const char* fft_wrapper_version(void);

#ifdef __cplusplus
}
#endif
