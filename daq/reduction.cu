#include "reduction.h"
#include <cuda.h>
#include <stdio.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) { 
        if (err != cudaSuccess) {
	          printf( "CUDA fail: %s in %s at line %d\n", cudaGetErrorString( err ),
			                  file, line );
		          exit(1);
			      }   
}
#define CHK( err ) (HandleError( err, __FILE__, __LINE__ ))

//Parallel reduction algorithm to calculate the mean of an array of numbers.
//Multiple adds are performed while loading to shared memory. 
//OPS_PER_THREAD specifies the number of elements to add while loading. 
//This reduces the number of blocks needed by a factor of OPS_PER_THREAD.
//Inputs: 
//        in: an array of floats. A separate average is computed for each 
//            gpu block. The number of elements must be a power of 2.
//        out: an array of floats to place the resulting averages. 
//             This must be the same size as the number of gpu blocks being used.
__global__ void mean_reduction (cufftReal * in, cufftReal * out){
        extern __shared__ cufftReal sdata[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x*OPS_PER_THREAD + threadIdx.x;
        sdata[tid] = 0;
        for(int j=0; j<OPS_PER_THREAD; j++)
	    sdata[tid]+=in[i+j*blockDim.x];
	__syncthreads();

        for(int s=blockDim.x/2; s>0; s>>=1){
                if(tid<s) sdata[tid] +=sdata[tid+s];
                __syncthreads();
        }   
	
	if(tid == 0) out[blockIdx.x] = sdata[0]/(blockDim.x*OPS_PER_THREAD);
}


//Parallel reduction algorithm to calculate the mean of squares for an array of numbers. 
//Multiple squarings and adds are performed while loading to shared memory. 
//OPS_PER_THREAD specifies the number of squares to add while loading. 
//This reduces the number of blocks needed by a factor of OPS_PER_THREAD.
//Inputs: 
//        in: an array of floats to average the squares of. A separate average 
//            is computed for each gpu block. The number of elements must be a power of 2.
//        out: an array of floats to place the resulting averages. This must be 
//            the same size as the number of gpu blocks being used in the kernel call.
__global__ void squares_reduction (cufftReal * in, cufftReal * out){
        extern __shared__ cufftReal sdata[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x*OPS_PER_THREAD + threadIdx.x;
        sdata[tid] = 0;
        for(int j=0; j<OPS_PER_THREAD; j++)
	    sdata[tid]+=in[i+j*blockDim.x] * in[i+j*blockDim.x];
        __syncthreads();
        for(int s =blockDim.x/2; s>0; s>>=1){
                if(tid<s) sdata[tid] +=sdata[tid+s];
                __syncthreads();
        }   

        if(tid == 0) out[blockIdx.x] = sdata[0]/(blockDim.x*OPS_PER_THREAD);
}

//Parallel reduction algorithm, to calculate the absolute max of an array of numbers
//Inputs: 
//        in: an array of floats. A separate max is computed for each gpu block. 
//            The number of elements must be a power of 2.
//        out: an array of floats to place the resulting maxes. This must be the 
//            same size as the number of gpu blocks being used in the kernel call.
__global__ void abs_max_reduction (cufftReal * in, cufftReal * out){
        extern __shared__ cufftReal sdata[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x*OPS_PER_THREAD + threadIdx.x;
        sdata[tid] = 0;
	for(int j =0; j<OPS_PER_THREAD; j++)
	    sdata[tid] = (abs(sdata[tid]) > abs(in[i+j*blockDim.x]))? abs(sdata[tid]): abs(in[i+j*blockDim.x]);
        __syncthreads();

        for(int s=blockDim.x/2; s>0; s>>=1){
                if(tid<s) sdata[tid] = (sdata[tid]> sdata[tid+s])? sdata[tid]:sdata[tid+s];
                __syncthreads();
        }
        if(tid == 0) out[blockIdx.x] = sdata[0];
}
	
//Calculate means of chunks of data by repeated kernel calls
//asynchronous with respect to host, so don't use output until actually finished
//Inputs: 
//        input: an  array of numbers to be divided into chunks, 
//               which will then be individually averaged
//	  deviceOutput: where the GPU writes the results to. 
//                The output array must contain  blockDim.x number of elements
//        output: host memory where the final results are written to. 
//               The number of elements in  this array must be n/chunkSize
//        n: the number of elements in the input aray
//        chunkSize: the number of elements per chunk
//        cs: the stream to use for the kernel calls
//        maxThreadsPerBlock: the maximum number of threads allowed per block 
//               (can be obtained by cudaGetDeviceProperties)
void getMeans(cufftReal *input, cufftReal * output, cufftReal * deviceOutput, int n, int chunkSize, cudaStream_t & cs, int maxThreadsPerBlock){
    if(chunkSize < OPS_PER_THREAD){
	printf("chunk size should not be smaller than OPS_PER_THREAD\n");
	exit(1);
    }
    int numThreads = min(maxThreadsPerBlock, chunkSize/OPS_PER_THREAD);
    int numBlocks = n/(numThreads * OPS_PER_THREAD);
    
    mean_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(input, deviceOutput);
    CHK(cudaGetLastError());
    int remaining = chunkSize/(numThreads*OPS_PER_THREAD); //number of threads, and number of summations that remain to be done per chunk
    while(remaining > 1){
	numThreads = min(maxThreadsPerBlock, remaining/OPS_PER_THREAD); 
	numBlocks = numBlocks/(numThreads*OPS_PER_THREAD);
	mean_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(deviceOutput, deviceOutput); //reusing input array for output!
	remaining/=(numThreads*OPS_PER_THREAD);
    }
    size_t memsize = numBlocks*sizeof(cufftReal); 
    CHK(cudaMemcpyAsync(output, deviceOutput, memsize, cudaMemcpyDeviceToHost, cs)); 
}


//Calculate means of squares of chunks of data by repeated kernel calls
//asynchronous with respect to host, so don't use output until actually finished
//Inputs: 
//        input: an array of numbers to be divided into chunks. 
//               For each chunk, the mean of the squares of the elements will be calculated.
//	  deviceOutput: where the GPU writes the results to. 
//               The output array must contain  blockDim.x number of elements
//        output: host memory where the final results are written to. 
//               The number of elements in  this array must be n/chunkSize
//        n: the number of elements in the input aray
//        chunkSize: the number of elements per chunk
//        cs: the stream to use for the kernel calls
//        maxThreadsPerBlock: the maximum number of threads allowed per block 
//               (can be obtained by cudaGetDeviceProperties)
void getMeansOfSquares(cufftReal *input, cufftReal * output, cufftReal * deviceOutput, int n, int chunkSize, cudaStream_t & cs, int maxThreadsPerBlock){
    if(chunkSize < OPS_PER_THREAD){
	printf("chunk size should not be smaller than OPS_PER_THREAD\n");
	exit(1);
    }
    int numThreads = min(maxThreadsPerBlock, chunkSize/OPS_PER_THREAD);
    int numBlocks = n/(numThreads * OPS_PER_THREAD);
    
    squares_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(input, deviceOutput);
    CHK(cudaGetLastError());
    
    int remaining = chunkSize/(numThreads*OPS_PER_THREAD); //number of threads, and number of summations that remain to be done
    while(remaining > 1){
	numThreads = min(maxThreadsPerBlock, remaining/OPS_PER_THREAD);
	numBlocks = numBlocks/(numThreads*OPS_PER_THREAD);
	mean_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(deviceOutput, deviceOutput); 
	remaining/=(numThreads*OPS_PER_THREAD);
    }
    size_t memsize = numBlocks*sizeof(cufftReal); 
    CHK(cudaMemcpyAsync(output, deviceOutput, memsize, cudaMemcpyDeviceToHost, cs)); 

}

//Calculate absolute maximum of chunks of data by repeated kernel calls
//asynchronous with respect to host, so don't use output until actually finished
//Inputs: 
//        input: an  array of numbers to be divided into chunks. 
//               For each chunk, the absolute maximum of the elements will be calculated.
//	  deviceOutput: where the GPU writes the results to. 
//               The output array must contain  blockDim.x number of elements
//        output: host memory where the final results are written to. 
//               The number of elements in  this array must be n/chunkSize
//        n: the number of elements in the input aray
//        chunkSize: the number of elements per chunk
//        cs: the stream to use for the kernel calls
//        maxThreadsPerBlock: the maximum number of threads allowed per block 
//               (can be obtained by cudaGetDeviceProperties)
void getAbsMax(cufftReal *input, cufftReal * output, cufftReal * deviceOutput, int n, int chunkSize, cudaStream_t & cs, int maxThreadsPerBlock){
    if(chunkSize < OPS_PER_THREAD){
	printf("chunk size should not be smaller than OPS_PER_THREAD\n");
	exit(1);
    }
    int numThreads = min(maxThreadsPerBlock, chunkSize/OPS_PER_THREAD);
    int numBlocks = n/(numThreads * OPS_PER_THREAD);
    abs_max_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(input, deviceOutput);
    CHK(cudaGetLastError());
    int remaining = chunkSize/(numThreads*OPS_PER_THREAD); //number of threads, and number of summations that remain to be done
    while(remaining > 1){
	numThreads = min(maxThreadsPerBlock, remaining/OPS_PER_THREAD);
	numBlocks = numBlocks/(numThreads*OPS_PER_THREAD);
	abs_max_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(deviceOutput, deviceOutput);
	remaining/=(numThreads*OPS_PER_THREAD);
    }
    size_t memsize = numBlocks*sizeof(cufftReal); 
    CHK(cudaMemcpyAsync(output, deviceOutput, memsize, cudaMemcpyDeviceToHost, cs));
}


//Reduction algorithm to calculate power spectrum. Take bsize complex numbers 
//starting at ffts[istart+bsize*blocknumber] and their copies in NCHUNS, and add the squares
//Input:
//      correction: corrects for the chunks that were nulled out because RFI. 
//                  It equals numChunks/(numChunks - numChunksNulled), where numChunksNulled is 
// the number of chunks nulled in this specific channel.
__global__ void ps_reduce(cufftComplex *ffts, float* output_ps, size_t istart, size_t avgsize, float correction) {
  int tid=threadIdx.x; // thread
  int bl=blockIdx.x; // block, ps bin #
  int nth=blockDim.x; //nthreads
  __shared__ float work[1024];
  //global pos
  size_t pos=istart+bl*avgsize;
  size_t pose=pos+avgsize;
  pos+=tid;
  work[tid]=0;
  while (pos<pose) {
    work[tid]+=ffts[pos].x*ffts[pos].x+ffts[pos].y*ffts[pos].y;
    pos+=nth;
  }
  // now do the tree reduce.
  int csum=nth/2;
  while (csum>0) {
    __syncthreads();
    if (tid<csum) {
      work[tid]+=work[tid+csum];
    }
    csum/=2;
  }
  if (tid==0) output_ps[bl]=work[0]*correction; //correcting for RFI
}


//Reduction algorithm, to calculate cross power spectrum
//Input:
//      correction: corrects for the chunks that were nulled out because RFI. 
//                  It equals numChunks/(numChunks - numChunksNulledCh1ORCH2)
__global__ void ps_X_reduce(cufftComplex *fftsA, cufftComplex *fftsB, 
			    float* output_ps_real, float* output_ps_imag, size_t istart, size_t avgsize, float correction) {
  int tid=threadIdx.x; // thread
  int bl=blockIdx.x; // block, ps bin #
  int nth=blockDim.x; //nthreads
  __shared__ float workR[1024];
  __shared__ float workI[1024];
  //global pos
  size_t pos=istart+bl*avgsize;
  size_t pose=pos+avgsize;
  pos+=tid;
  workR[tid]=0;
  workI[tid]=0;
  while (pos<pose) {
    workR[tid]+=fftsA[pos].x*fftsB[pos].x+fftsA[pos].y*fftsB[pos].y;
    workI[tid]+=fftsA[pos].x*fftsB[pos].y-fftsA[pos].y*fftsB[pos].x;
    pos+=nth;
  }
  // now do the tree reduce.
  int csum=nth/2;
  while (csum>0) {
    __syncthreads();
    if (tid<csum) {
      workR[tid]+=workR[tid+csum];
      workI[tid]+=workI[tid+csum];
    }
    csum/=2;
  }
  if (tid==0) {
    output_ps_real[bl]=workR[0]*correction;
    output_ps_imag[bl]=workI[0]*correction;
  }
} 