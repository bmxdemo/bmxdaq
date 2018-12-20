//Parallel reduction algorithms to calculate different statistics over large arrays of numbers
//for RFI detection and computing spectra
#pragma once
#include <cufft.h>
#define OPS_PER_THREAD 8  //do OPS_PER_THREAD operations while loading into shared memory, to reduce the number of blocks needed
                          //must be power of 2!


//Calculate power spectrum and bin frequencies
__global__ void ps_reduce(cufftComplex *ffts, float* output_ps, size_t istart, size_t avgsize);

//Calculate cross power spectrum and bin frequencies
__global__ void ps_X_reduce(cufftComplex *fftsA, cufftComplex *fftsB, float* output_ps_real, 
	                    float* output_ps_imag, size_t istart, size_t avgsize);
