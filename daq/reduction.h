//Parallel reduction algorithms to calculate different statistics over large arrays of numbers
//for RFI detection and computing spectra
#include <cufft.h>
#define OPS_PER_THREAD 8  //do OPS_PER_THREAD operations while loading into shared memory, to reduce the number of blocks needed
                          //must be power of 2!

//Calculate means
__global__ void mean_reduction (cufftReal * in, cufftReal * out);

//Calculate means of squares
__global__ void squares_reduction (cufftReal * in, cufftReal * out);

//Calculate abosulute maximums 
__global__ void abs_max_reduction (cufftReal * in, cufftReal * out);



//Recursive calls to mean_reduction to calculate means for each chunk
void getMeans(cufftReal *input, cufftReal * output, cufftReal * deviceOutput, 
	      int n, int chunkSize, cudaStream_t & cs, int maxThreadsPerBlock);

//Recursive calls to mean_reduction to calculate means of squares for each chunk
void getMeansOfSquares(cufftReal *input, cufftReal * output, cufftReal * deviceOutput, 
	               int n, int chunkSize, cudaStream_t & cs, int maxThreadsPerBlock);

//Recursive calls to mean_reduction to calculate absolute maximums for each chunk
void getAbsMax(cufftReal *input, cufftReal * output, cufftReal * deviceOutput, 
	       int n, int chunkSize, cudaStream_t & cs, int maxThreadsPerBlock);



//Calculate power spectrum and bin frequencies
__global__ void ps_reduce(cufftComplex *ffts, float* output_ps, size_t istart, size_t avgsize, float correction);

//Calculate cross power spectrum and bin frequencies
__global__ void ps_X_reduce(cufftComplex *fftsA, cufftComplex *fftsB, float* output_ps_real, 
	                    float* output_ps_imag, size_t istart, size_t avgsize, float correction);
