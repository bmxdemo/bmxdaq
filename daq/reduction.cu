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
//Inputs: 
//        in: an array of floats. A separate average is computed for each 
//            gpu block. The number of elements must be a power of 2.
//        out: an array of floats to place the resulting averages. 
//             This must be the same size as the number of gpu blocks being used.
//        opsPerThread: number of elements to add while loading. This reduces
//             the number of blocks needed by a factor of opsPerThread.
__global__ void mean_reduction (cufftReal * in, cufftReal * out, int opsPerThread){
        extern __shared__ cufftReal sdata[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x*opsPerThread + threadIdx.x;
        sdata[tid] = 0;
        for(int j=0; j<opsPerThread; j++)
	    sdata[tid]+=in[i+j*blockDim.x];
	__syncthreads();

        for(int s=blockDim.x/2; s>0; s>>=1){
                if(tid<s) sdata[tid] +=sdata[tid+s];
                __syncthreads();
        }   
	
	if(tid == 0) out[blockIdx.x] = sdata[0]/(blockDim.x*opsPerThread);
}


//Parallel reduction algorithm to calculate the mean of squares for an array of numbers. 
//Multiple squarings and adds are performed while loading to shared memory. 
//Inputs: 
//        in: an array of floats to average the squares of. A separate average 
//            is computed for each gpu block. The number of elements must be a power of 2.
//        out: an array of floats to place the resulting averages. This must be 
//            the same size as the number of gpu blocks being used in the kernel call.
//        opsPerThread: number of elements to add while loading. This reduces
//             the number of blocks needed by a factor of opsPerThread.
__global__ void squares_reduction (cufftReal * in, cufftReal * out, int opsPerThread){
        extern __shared__ cufftReal sdata[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x*opsPerThread + threadIdx.x;
        sdata[tid] = 0;
        for(int j=0; j<opsPerThread; j++)
	    sdata[tid]+=in[i+j*blockDim.x] * in[i+j*blockDim.x];
        __syncthreads();
        for(int s =blockDim.x/2; s>0; s>>=1){
                if(tid<s) sdata[tid] +=sdata[tid+s];
                __syncthreads();
        }   

        if(tid == 0) out[blockIdx.x] = sdata[0]/(blockDim.x*opsPerThread);
}

//Parallel reduction algorithm, to calculate the absolute max of an array of numbers
//Inputs: 
//        in: an array of floats. A separate max is computed for each gpu block. 
//            The number of elements must be a power of 2.
//        out: an array of floats to place the resulting maxes. This must be the 
//            same size as the number of gpu blocks being used in the kernel call.
//        opsPerThread: number of elements to add while loading. This reduces
//             the number of blocks needed by a factor of opsPerThread.
__global__ void abs_max_reduction (cufftReal * in, cufftReal * out, int opsPerThread){
        extern __shared__ cufftReal sdata[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x*opsPerThread + threadIdx.x;
        sdata[tid] = 0;
	for(int j =0; j<opsPerThread; j++)
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
    int opsPerThread = min(chunkSize, OPS_PER_THREAD);
    int numThreads = min(maxThreadsPerBlock, chunkSize/opsPerThread);
    int numBlocks = n/(numThreads * opsPerThread);
    mean_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(input, deviceOutput, opsPerThread);
    
    int remaining = chunkSize/(numThreads*opsPerThread); //number of threads, and number of summations that remain to be done per chunk
    while(remaining > 1){
	opsPerThread = min(remaining, opsPerThread);
	numThreads = min(maxThreadsPerBlock, remaining/opsPerThread); 
	numBlocks = numBlocks/(numThreads*opsPerThread);
	mean_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(deviceOutput, deviceOutput, opsPerThread); //reusing input array for output!
	remaining/=(numThreads*opsPerThread);
    }
    CHK(cudaGetLastError());
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
    int opsPerThread = min(chunkSize, OPS_PER_THREAD);
    int numThreads = min(maxThreadsPerBlock, chunkSize/opsPerThread);
    int numBlocks = n/(numThreads * opsPerThread);
    
    squares_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(input, deviceOutput, opsPerThread);
    
    int remaining = chunkSize/(numThreads*opsPerThread); //number of threads, and number of summations that remain to be done
    while(remaining > 1){
	opsPerThread = min(remaining, opsPerThread);
	numThreads = min(maxThreadsPerBlock, remaining/opsPerThread);
	numBlocks = numBlocks/(numThreads*opsPerThread);
	mean_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(deviceOutput, deviceOutput, opsPerThread); 
	remaining/=(numThreads*opsPerThread);
    }
    CHK(cudaGetLastError());
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
    int opsPerThread = min(chunkSize, OPS_PER_THREAD);
    int numThreads = min(maxThreadsPerBlock, chunkSize/opsPerThread);
    int numBlocks = n/(numThreads * opsPerThread);
    
    abs_max_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(input, deviceOutput, opsPerThread);
    
    int remaining = chunkSize/(numThreads*opsPerThread); //number of threads, and number of summations that remain to be done
    while(remaining > 1){
	opsPerThread = min(remaining, opsPerThread);
	numThreads = min(maxThreadsPerBlock, remaining/opsPerThread);
	numBlocks = numBlocks/(numThreads*opsPerThread);
	abs_max_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(deviceOutput, deviceOutput, opsPerThread);
	remaining/=(numThreads*opsPerThread);
    }
    CHK(cudaGetLastError());
    size_t memsize = numBlocks*sizeof(cufftReal); 
    CHK(cudaMemcpyAsync(output, deviceOutput, memsize, cudaMemcpyDeviceToHost, cs));
}


//Reduction algorithm to calculate power spectrum. Take bsize complex numbers 
//starting at ffts[istart+bsize*blocknumber] and their copies in NCHUNS, and add the squares
__global__ void ps_reduce(cufftComplex *ffts, float* output_ps, size_t istart, size_t avgsize) {
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
  if (tid==0) output_ps[bl]=work[0];
}

//Reduction algorithm, to calculate cross power spectrum
__global__ void ps_X_reduce(cufftComplex *fftsA, cufftComplex *fftsB, 
			    float* output_ps_real, float* output_ps_imag, size_t istart, size_t avgsize) {
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
    output_ps_real[bl]=workR[0];
    output_ps_imag[bl]=workI[0];
  }
} 





// Overwrite CH1 FFT with (CH1+CH2)*CONJ((CH3+CH4))
__global__ void C12_Cross(cufftComplex *ffts1, cufftComplex *ffts2, cufftComplex *ffts3, cufftComplex *ffts4) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x);
  // float ch12r=ffts1[i].x;//+ffts2[i].x;
  // float ch12i=ffts1[i].y;//#+ffts2[i].y;
  // float phi=-0.5e0*i;
  // float cphi=cos(phi);
  // float sphi=sin(phi);
  // float ch34r=ch12r*cphi-ch12i*sphi;
  // float ch34i=ch12r*sphi+ch12i*cphi;

  float ch12r=ffts1[i].x+ffts2[i].x;
  float ch12i=ffts1[i].y+ffts2[i].y;
  float ch34r=ffts3[i].x+ffts4[i].x;
  float ch34i=ffts3[i].y+ffts4[i].y;

  float XR=(ch12r*ch34r)+(ch12i*ch34i);
  float XI=(ch12r*ch34i)-(ch12i*ch34r);
  
  // float ch12r=ffts1[i].x;
  // float ch12i=ffts1[i].y;
  // float ch34r=ffts2[i].x;
  // float ch34i=ffts2[i].y;

  ffts1[i].x=XR;
  ffts1[i].y=XI;
}


// find a global maximum and store it into device variable
__global__ void C12_FindMax(cufftReal *data, int totsize, int* output) {
  int tid=threadIdx.x; // thread
  //int bl=blockIdx.x; // block, ps bin # // assumed zero here
  int nth=blockDim.x; //nthreads
  __shared__ float work[1024];
  __shared__ int iwork[1024];
  //global pos
  size_t pos=tid;
  work[tid]=-1e99;
  iwork[tid]=-10;
  while (pos<totsize) {
    if (data[pos]>work[tid]) {
      work[tid]=data[pos];
      iwork[tid]=pos;
    }
    pos+=nth;
  }
  // now do the tree reduce.
  int csum=nth/2;
  while (csum>0) {
    __syncthreads();
    if (tid<csum) {
      if (work[tid+csum]>work[tid]) {
	work[tid]=work[tid+csum];
	iwork[tid]=iwork[tid+csum];
      }
    }
    csum/=2;
  }
  if (tid==0) {
      int res=iwork[0];
      if (res>totsize/2) res-=totsize;
      *output=res;
    }
}




