/***********************************
***********************************
CUDA PART
***********************************
**********************************/

#define CUDA_COMPILE //to enable cuda types in gpucard.h
#include "gpucard.h"
#undef CUDA_COMPILE
#include "terminal.h"

#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <time.h>
#define FLOATIZE_X 2
#define OPS_PER_THREAD 8 //for parallel reduction algorithms: do OPS_PER_THREAD operations while loading into shared memory, to reduce the number of blocks needed
			 //must be power of 2!



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


//Initialize instance of GPUCARD
//Input:
//      gc: instance of GPUCARD to initialize
//      set: settings
void gpuCardInit (GPUCARD *gc, SETTINGS *set) {
  
  //print out gpu device properties
    gc->devProp = (cudaDeviceProp *)malloc(sizeof(cudaDeviceProp));
    CHK(cudaGetDeviceProperties(gc->devProp, 0));
  printf("\nGPU properties \n====================\n");
  printf("Version number:                %d.%d\n",  gc->devProp->major, gc->devProp->minor);
  printf("Name:                          %s\n",  gc->devProp->name);
  printf("Total global memory:           %u\n",  gc->devProp->totalGlobalMem);
  printf("Total shared memory per block: %u\n",  gc->devProp->sharedMemPerBlock);
  printf("Total registers per block:     %d\n",  gc->devProp->regsPerBlock);
  printf("Warp size:                     %d\n",  gc->devProp->warpSize);
  printf("Maximum memory pitch:          %u\n",  gc->devProp->memPitch);
  printf("Maximum threads per block:     %d\n",  gc->devProp->maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
  printf("Maximum dimension %d of block:  %d\n", i, gc->devProp->maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
  printf("Maximum dimension %d of grid:   %d\n", i, gc->devProp->maxGridSize[i]);
  printf("Clock rate:                    %d\n",  gc->devProp->clockRate);
  printf("Total constant memory:         %u\n",  gc->devProp->totalConstMem);
  printf("Texture alignment:             %u\n",  gc->devProp->textureAlignment);
  printf("Concurrent copy and execution: %s\n",  (gc->devProp->deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n",  gc->devProp->multiProcessorCount);
  printf("Kernel execution timeout:      %s\n\n",  (gc->devProp->kernelExecTimeoutEnabled ? "Yes" : "No"));

  printf ("\n\nInitializing GPU\n");
  printf ("====================\n");
  printf ("Allocating GPU buffers\n");
  int Nb=set->cuda_streams;
  gc->cbuf=(int8_t**)malloc(Nb*sizeof(int8_t*));
  gc->cfbuf=(cufftReal**)malloc(Nb*sizeof(cufftReal*));
  gc->cfft=(cufftComplex**)malloc(Nb*sizeof(cufftComplex*));
  gc->coutps=(float**)malloc(Nb*sizeof(float*));
  int nchan=gc->nchan=1+(set->channel_mask==3);
  if ((nchan==2) and (FLOATIZE_X%2==1)) {
    printf ("Need FLOATIZE_X even for two channels\n");
    exit(1);
  }
  if(!(OPS_PER_THREAD>0) ||  !((OPS_PER_THREAD & (OPS_PER_THREAD-1)) == 0)){
      printf("Need OPS_PER_THREAD to be a power of 2.\n");
      exit(1);
  }

  gc->fftsize=set->fft_size;
  uint32_t bufsize=gc->bufsize=set->fft_size*nchan;
  uint32_t transform_size=(set->fft_size/2+1)*nchan;
  float nunyq=set->sample_rate/2;
  float dnu=nunyq/(set->fft_size/2+1);
  gc->tot_pssize=0;
  gc->ncuts=set->n_cuts;
  for (int i=0; i<gc->ncuts; i++) {
    printf ("Cutout %i:\n",i);
    gc->fftavg[i]=set->fft_avg[i];
    // first sort  reflections etc.
    float numin, numax;
    numin=set->nu_min[i];
    numax=set->nu_max[i];
    while (fabs(numin)>nunyq) numin-=set->sample_rate;
    while (fabs(numax)>nunyq) numax-=set->sample_rate;
    numin=abs(numin);
    numax=abs(numax);
    if (numax<numin) { float t=numin; numin=numax; numax=t; }
    printf ("   Frequencies %f - %f Mhz appear as %f - %f \n",set->nu_min[i]/1e6, set->nu_max[i]/1e6,
	    numin/1e6, numax/1e6);
    int imin=int(numin/dnu);
    if (imin==0) imin=1;
    int imax=int(numax/dnu)+1;
    gc->pssize1[i]=(imax-imin)/set->fft_avg[i];
    gc->ndxofs[i]=imin;
    if ((imax-imin)%set->fft_avg[i]>0) gc->pssize1[i]+=1;
    imax=imin+gc->pssize1[i]*set->fft_avg[i];
    numin=imin*dnu;
    numax=imax*dnu;
    set->nu_min[i]=numin;
    set->nu_max[i]=numax;
    set->pssize[i]=gc->pssize1[i];
    if (nchan==2)
      gc->pssize[i]=gc->pssize1[i]*4; // for two channels and two crosses
    else
      gc->pssize[i]=gc->pssize1[i]; // just one power spectrum
    gc->tot_pssize+=gc->pssize[i];
    printf ("   Actual freq range: %f - %f MHz (edges!)\n",numin/1e6, numax/1e6);
    printf ("   # PS offset, #PS bins: %i %i\n",gc->ndxofs[i],gc->pssize1[i]);
  }
  for (int i=0;i<Nb;i++) {
    CHK(cudaMalloc(&gc->cbuf[i],bufsize));
    CHK(cudaMalloc(&gc->cfbuf[i], bufsize*sizeof(cufftReal)));
    CHK(cudaMalloc(&gc->cfft[i],transform_size*sizeof(cufftComplex)));
    CHK(cudaMalloc(&gc->coutps[i],gc->tot_pssize*sizeof(float)));
  }
  CHK(cudaHostAlloc(&gc->outps, gc->tot_pssize*sizeof(float), cudaHostAllocDefault));

  printf ("Setting up CUFFT\n");
  int status=cufftPlanMany(&gc->plan, 1, (int*)&(set->fft_size), NULL, 0, 0, 
        NULL, transform_size,1, CUFFT_R2C, nchan);

  if (status!=CUFFT_SUCCESS) {
       printf ("Plan failed:");
       if (status==CUFFT_ALLOC_FAILED) printf("CUFFT_ALLOC_FAILED");
       if (status==CUFFT_INVALID_VALUE) printf ("CUFFT_INVALID_VALUE");
       if (status==CUFFT_INTERNAL_ERROR) printf ("CUFFT_INTERNAL_ERROR");
       if (status==CUFFT_SETUP_FAILED) printf ("CUFFT_SETUP_FAILED");
       if (status==CUFFT_INVALID_SIZE) printf ("CUFFT_INVALID_SIZE");
       printf("\n");
       exit(1);
  }
  printf ("Setting up CUDA streams & events\n");
  gc->nstreams = set->cuda_streams;
  gc->threads=set->cuda_threads;
  if (gc->nstreams<1) {
    printf ("Cannot really work with less than one stream.\n");
    exit(1);
  }
  gc->streams=(cudaStream_t*)malloc(gc->nstreams*sizeof(cudaStream_t));
  gc->eStart=(cudaEvent_t*)malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneCopy=(cudaEvent_t*)malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneFloatize=(cudaEvent_t*)malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneFFT=(cudaEvent_t*)malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDonePost=(cudaEvent_t*)malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneCopyBack=(cudaEvent_t*)malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneRFI=(cudaEvent_t*)malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eBeginCopyBack=(cudaEvent_t*)malloc(gc->nstreams*sizeof(cudaEvent_t));
  for (int i=0;i<gc->nstreams;i++) {
    //create stream
    CHK(cudaStreamCreate(&gc->streams[i]));
    //create events for stream
    CHK(cudaEventCreate(&gc->eStart[i]));
    CHK(cudaEventCreate(&gc->eDoneCopy[i]));
    CHK(cudaEventCreate(&gc->eDoneFloatize[i]));
    CHK(cudaEventCreate(&gc->eDoneFFT[i]));
    CHK(cudaEventCreate(&gc->eDonePost[i]));
    CHK(cudaEventCreate(&gc->eDoneCopyBack[i]));
    CHK(cudaEventCreate(&gc->eDoneRFI[i]));
    CHK(cudaEventCreate(&gc->eBeginCopyBack[i]));
  }
 
  gc->fstream = 0; //oldest running stream
  gc->bstream = -1; //newest stream
  gc->active_streams = 0; //number of streams currently running
  
  //allocate memory for RFI statistics
  gc->chunkSize = pow(2,set->log_chunk_size);
  int numThreads = min(gc->devProp->maxThreadsPerBlock, gc->chunkSize/OPS_PER_THREAD);
  int numBlocks = gc->bufsize/(numThreads * OPS_PER_THREAD);  //number of blocks needed for first kernel call in parallel reduction algorithms
  
  gc->mean = (cufftReal **)malloc(Nb*sizeof(cufftReal*));
  gc->cmean = (cufftReal **) malloc(Nb *sizeof(cufftReal*));
  gc->sqMean = (cufftReal **)malloc(Nb*sizeof(cufftReal*));
  gc->csqMean = (cufftReal **) malloc(Nb *sizeof(cufftReal*));
  gc->variance = (cufftReal **)malloc(Nb*sizeof(cufftReal*));
  gc->absMax = (cufftReal **) malloc(Nb*sizeof(cufftReal*));
  gc->cabsMax = (cufftReal **) malloc(Nb*sizeof(cufftReal*));
  gc->isOutlier = (int **)malloc(Nb*sizeof(int*));
  gc->outlierBuf = (int8_t * )malloc(gc->chunkSize);
  
  for(int i=0; i<Nb; i++){
      CHK(cudaMalloc(&gc->cmean[i], numBlocks*sizeof(cufftReal)));
      CHK(cudaMallocHost(&gc->mean[i], gc->bufsize/gc->chunkSize*sizeof(cufftReal)));
      CHK(cudaMalloc(&gc->csqMean[i], numBlocks*sizeof(cufftReal)));
      CHK(cudaMallocHost(&gc->sqMean[i], gc->bufsize/gc->chunkSize*sizeof(cufftReal)));
      CHK(cudaMallocHost(&gc->variance[i], gc->bufsize/gc->chunkSize*sizeof(cufftReal))); //total number of chunks in all channels
      CHK(cudaMalloc(&gc->cabsMax[i], numBlocks*sizeof(cufftReal)));
      CHK(cudaMallocHost(&gc->absMax[i], gc->bufsize/gc->chunkSize*sizeof(cufftReal)));
      gc->isOutlier[i] = (int *)malloc(gc->bufsize/gc->chunkSize/gc->nchan * sizeof(int)); //number of chunks in 1 channel
  }

  gc->avgOutliersPerChannel = (float *)malloc(gc->nchan*sizeof(float));
  memset(gc->avgOutliersPerChannel, 0, gc->nchan*sizeof(float));
  printf ("GPU ready.\n");

}

//Convert bytes to floats, 1 channel version
//Inputs:
//	 sample: array of bytes
//       fsample: array of floats to put output in
__global__ void floatize_1chan(int8_t* sample, cufftReal* fsample)  {
    int i = FLOATIZE_X*(blockDim.x * blockIdx.x + threadIdx.x);
    for (int j=0; j<FLOATIZE_X; j++) fsample[i+j]=float(sample[i+j]);
}


//Convert bytes to floats, 2 channel version
//Inputs:
//	 sample: array of bytes with the 2 channels interleaved
//       fsample1: array of floats to put converted bytes from channel 1 in
//       fsample2: array of floats to put converted bytes from channel 2 in
__global__ void floatize_2chan(int8_t* sample, cufftReal* fsample1, cufftReal* fsample2)  {
      int i = FLOATIZE_X*(blockDim.x * blockIdx.x + threadIdx.x);
      for (int j=0; j<FLOATIZE_X/2; j++) {
      fsample1[i/2+j]=float(sample[i+2*j]);
      fsample2[i/2+j]=float(sample[i+2*j+1]);
    }
}


//Parallel reduction algorithm to calculate the mean of an array of numbers. Multiple adds are performed while loading to shared memory. 
//OPS_PER_THREAD specifies the number of elements to add while loading. This reduces the number of blocks needed by a factor of OPS_PER_THREAD.
//Inputs: 
//        in: an array of floats to average. A separate average is computed for each gpu block. The number of elements must be a power of 2.
//        out: an array of floats to place the resulting averages. This must be the same size as the number of gpu blocks being used in the kernel call.
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


//Parallel reduction algorithm to calculate the mean of squares for an array of numbers. Multiple squarings and adds are performed while loading to shared memory. 
//OPS_PER_THREAD specifies the number of squares to add while loading. This reduces the number of blocks needed by a factor of OPS_PER_THREAD.
//Inputs: 
//        in: an array of floats to average the squares of. A separate average is computed for each gpu block. The number of elements must be a power of 2.
//        out: an array of floats to place the resulting averages. This must be the same size as the number of gpu blocks being used in the kernel call.
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
//        in: an array of floats. A separate max is computed for each gpu block. The number of elements must be a power of 2.
//        out: an array of floats to place the resulting maxes. This must be the same size as the number of gpu blocks being used in the kernel call.
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
//Inputs: 
//        input: an  array of numbers to be divided into chunks, which will then be individually averaged
//	  deviceOutput: where the GPU writes the results to. The output array must contain  blockDim.x number of elements
//        output: host memory where the final results are written to. The number of elements in  this array must be n/chunkSize
//        n: the number of elements in the input aray
//        chunkSize: the number of elements per chunk
//        cs: the stream to use for the kernel calls
//        maxThreadsPerBlock: the maximum number of threads allowed per block (can be obtained by cudaGetDeviceProperties)
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
//Inputs: 
//        input: an  array of numbers to be divided into chunks. For each chunk, the mean of the squares of the elements will be calculated.
//	  deviceOutput: where the GPU writes the results to. The output array must contain  blockDim.x number of elements
//        output: host memory where the final results are written to. The number of elements in  this array must be n/chunkSize
//        n: the number of elements in the input aray
//        chunkSize: the number of elements per chunk
//        cs: the stream to use for the kernel calls
//        maxThreadsPerBlock: the maximum number of threads allowed per block (can be obtained by cudaGetDeviceProperties)
void getMeansOfSquares(cufftReal *input, cufftReal * output, cufftReal * deviceOutput, int n, int chunkSize, cudaStream_t & cs, int maxThreadsPerBlock){
    if(chunkSize < OPS_PER_THREAD){
	printf("chunk size should not be smaller than OPS_PER_THREAD\n");
	exit(1);
    }
    int numThreads = min(maxThreadsPerBlock, chunkSize/OPS_PER_THREAD);
    int numBlocks = n/(numThreads * OPS_PER_THREAD);
    //printf("n is : %d\n", n);
    squares_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(input, deviceOutput);
    CHK(cudaGetLastError());
    int remaining = chunkSize/(numThreads*OPS_PER_THREAD); //number of threads, and number of summations that remain to be done
    //printf("remaining : %d\nnumThreads: %d\nnumBlocks: %d\n",remaining,  0, numBlocks);
    while(remaining > 1){
	numThreads = min(maxThreadsPerBlock, remaining/OPS_PER_THREAD);
	numBlocks = numBlocks/(numThreads*OPS_PER_THREAD);

	//printf("remaining : %d\nnumThreads: %d\nnumBlocks: %d\n",remaining,  numThreads, numBlocks);

	mean_reduction<<<numBlocks, numThreads, numThreads*sizeof(cufftReal), cs>>>(deviceOutput, deviceOutput); 
	remaining/=(numThreads*OPS_PER_THREAD);
    }
    size_t memsize = numBlocks*sizeof(cufftReal); 
    CHK(cudaMemcpyAsync(output, deviceOutput, memsize, cudaMemcpyDeviceToHost, cs)); //need synchronous copying so that cpu blocks and doesn't use this memory until finished copyingi

}

//Calculate absolute maximum of chunks of data by repeated kernel calls
//Inputs: 
//        input: an  array of numbers to be divided into chunks. For each chunk, the absolute maximum of the elements will be calculated.
//	  deviceOutput: where the GPU writes the results to. The output array must contain  blockDim.x number of elements
//        output: host memory where the final results are written to. The number of elements in  this array must be n/chunkSize
//        n: the number of elements in the input aray
//        chunkSize: the number of elements per chunk
//        cs: the stream to use for the kernel calls
//        maxThreadsPerBlock: the maximum number of threads allowed per block (can be obtained by cudaGetDeviceProperties)
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

cufftDoubleReal variance(cufftDoubleReal ssquare, cufftDoubleReal mean){
    return ssquare - pow(mean, 2);
}


//Reduction algorithm to calculate power spectrum. Take bsize complex numbers starting at ffts[istart+bsize*blocknumber]
//and their copies in NCHUNS, and add the squares
//Input:
//      correction: corrects for the chunks that were nulled out because RFI. It equaks numChunks/(numChunks - numChunksNulled), where numChunksNulled is 
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


//Reduction algorith, to calculate cross power spectrum
//Input:
//      correction: corrects for the chunks that were nulled out because RFI. It equaks numChunks/(numChunks - numChunksNulledCh1ORCH2)
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

//Print the elapsed time between 2 cuda events
void printDt (cudaEvent_t cstart, cudaEvent_t cstop, float & total) {
  float gpu_time;
  CHK(cudaEventElapsedTime(&gpu_time, cstart, cstop));
  printf (" %3.2fms ", gpu_time);
  total +=gpu_time;
}

void printTiming(GPUCARD *gc, int i) {
  float totalTime = 0;
  printf ("GPU timing (copy/floatize/RFI/fft/post/copyback): ");
  printDt (gc->eStart[i], gc->eDoneCopy[i], totalTime);
  printDt (gc->eDoneCopy[i], gc->eDoneFloatize[i], totalTime);
  printDt (gc->eDoneFloatize[i], gc->eDoneRFI[i], totalTime);
  printDt (gc->eDoneRFI[i], gc->eDoneFFT[i], totalTime);
  printDt (gc->eDoneFFT[i], gc->eDonePost[i], totalTime);
  printDt (gc->eBeginCopyBack[i], gc->eDoneCopyBack[i], totalTime);
  tprintfn (" total: %3.2f ", totalTime);
}


//Process one data packet from the digitizer
//Input:
//	gc: graphics card
//      buf: data from digitizer
//      wr: writer to write out power spectra and outliers to files
//	set: settings
bool gpuProcessBuffer(GPUCARD *gc, int8_t *buf, WRITER *wr, SETTINGS *set) {
    //streamed version
    bool deleteLinesInConsole = false;
   //Check if other streams are finished and proccess the finished ones in order (i.e. print output to file)
   while(gc->active_streams > 0){
	if(cudaEventQuery(gc->eDonePost[gc->fstream])==cudaSuccess){
                if(deleteLinesInConsole) 
		    for(int i=0; i<4; i++){
			printf("\33[2K"); //delete line in console
			treturn(1); //move cursor up a line;
		    }
	        //print time and write to file
                cudaEventRecord(gc->eBeginCopyBack[gc->fstream], gc->streams[gc->fstream]);
		CHK(cudaMemcpyAsync(gc->outps,gc->coutps[gc->fstream], gc->tot_pssize*sizeof(float), cudaMemcpyDeviceToHost, gc->streams[gc->fstream]));
                cudaEventRecord(gc->eDoneCopyBack[gc->fstream], gc->streams[gc->fstream]);
                //cudaThreadSynchronize();
		cudaEventSynchronize(gc->eDoneCopyBack[gc->fstream]);
		printTiming(gc,gc->fstream);
                if (set->print_meanvar) {
                  // now find some statistic over subsamples of samples
                  uint32_t bs=gc->bufsize;
                  uint32_t step=gc->bufsize/(32768);
                  float NSub=bs/step; // number of subsamples to take
                  float m1=0.,m2=0.,v1=0.,v2=0.;
                  for (int i=0; i<bs; i+=step) { // take them in steps of step
                 	float n=buf[i];
                	m1+=n; v1+=n*n;
                	n=buf[i+1];
                	m2+=n; v2+=n*n;
                  }
                  m1/=NSub; v1=sqrt(v1/NSub-m1*m1); //mean and variance
                  m2/=NSub; v2=sqrt(v2/NSub-m2*m2);
                  tprintfn ("CH1 mean/rms: %f %f   CH2 mean/rms: %f %f   ",m1,v1,m2,v2);
                }
                if (set->print_maxp) {
                  // find max power in each cutout in each channel.
                  int of1=0; // CH1 auto
                  for (int i=0; i<gc->ncuts; i++) {
            	float ch1p=0, ch2p=0;
            	int ch1i=0, ch2i=0;
            	int of2=of1+gc->pssize1[i]; //CH2 auto
            	for (int j=0; j<gc->pssize1[i];j++) {
            	  if (gc->outps[of1+j] > ch1p) {ch1p=gc->outps[of1+j]; ch1i=j;}
            	  if (gc->outps[of2+j] > ch2p) {ch2p=gc->outps[of2+j]; ch2i=j;}
            	}
            	of1+=gc->pssize[i];  // next cutout 
            	float numin=set->nu_min[i];
            	float nustep=(set->nu_max[i]-set->nu_min[i])/(gc->pssize1[i]);
            	float ch1f=(numin+nustep*(0.5+ch1i))/1e6;
            	float ch2f=(numin+nustep*(0.5+ch2i))/1e6;
            	tprintfn ("Peak pow (cutout %i): CH1 %f at %f MHz;   CH2 %f at %f MHz  ",i,log(ch1p),ch1f,log(ch2p),ch2f);
                  }
                }
                writerWritePS(wr,gc->outps);
        	gc->fstream = (++gc->fstream)%(gc->nstreams);
                gc->active_streams--;
		
		deleteLinesInConsole =  true;

 	}
        else 
		break;
        
    }

    
    int csi = gc->bstream = (++gc->bstream)%(gc->nstreams); //add new stream

    if(gc->active_streams == gc->nstreams){ //if no empty streams
    	printf("No free streams.\n");
        if(gc->nstreams > 1) //first few packets come in close together (<122 ms), so for 1 stream, we need to queue them, and not just quit the program
		exit(1);
    }

    gc->active_streams++;

    cudaStream_t cs= gc->streams[gc->bstream];
    cudaEventRecord(gc->eStart[csi], cs);

    CHK(cudaMemcpyAsync(gc->cbuf[csi], buf, gc->bufsize , cudaMemcpyHostToDevice,cs));
    
    cudaEventRecord(gc->eDoneCopy[csi], cs);
    int threadsPerBlock = gc->threads;
    int blocksPerGrid = gc->bufsize / threadsPerBlock/FLOATIZE_X;
    if (gc->nchan==1) 
      floatize_1chan<<<blocksPerGrid, threadsPerBlock, 0, cs>>>(gc->cbuf[csi],gc->cfbuf[csi]);
    else 
      floatize_2chan<<<blocksPerGrid, threadsPerBlock, 0, cs>>>(gc->cbuf[csi],gc->cfbuf[csi],&(gc->cfbuf[csi][gc->fftsize]));
    cudaEventRecord(gc->eDoneFloatize[csi], cs);
    

    struct timespec start, now;
    clock_gettime(CLOCK_REALTIME, &start);

    //RFI rejection       
    int numChunks = gc->bufsize/gc->chunkSize; //total number of chunks in all channels
    int o[2] ={0}; //number of outliers per channel
    int outliersOR = 0; //number of outliers obtained by a logical OR on the arrays of outlier flags from the different channels
    
    if(gc->nchan==2){//so far specific to 2 channels. Can generalize when know data format of 3 or 4 channels
	getMeans(gc->cfbuf[csi], gc->mean[csi], gc->cmean[csi], gc->bufsize, gc->chunkSize, cs, gc->devProp->maxThreadsPerBlock); 
	getMeansOfSquares(gc->cfbuf[csi], gc->sqMean[csi], gc->csqMean[csi], gc->bufsize, gc->chunkSize, cs, gc->devProp->maxThreadsPerBlock); 
	getAbsMax(gc->cfbuf[csi], gc->absMax[csi], gc->cabsMax[csi], gc->bufsize, gc->chunkSize, cs, gc->devProp->maxThreadsPerBlock); 

        clock_gettime(CLOCK_REALTIME, &now);
	//tprintfn("Time after kernel calls: %f", (now.tv_sec-start.tv_sec)+(now.tv_nsec - start.tv_nsec)/1e9);


	cufftDoubleReal tmean[2]={0}, tsqMean[2]={0}, tvar[2], trms[2]; //for 2 channels. Note: double precision is neccesary or results will be incorrect!
	memset(gc->isOutlier[csi], 0, numChunks/gc->nchan*sizeof(int)); //reset outlier flags to 0
	
	cufftReal ** statistic = gc->variance; //desired statistic(s) to use to determine outliers
        
	//synchronize so don't use memory before GPU finishes copying it to the CPU
        CHK(cudaStreamSynchronize(cs));

        clock_gettime(CLOCK_REALTIME, &now);
	//tprintfn("Time after synchronize stream: %f", (now.tv_sec-start.tv_sec)+(now.tv_nsec - start.tv_nsec)/1e9);
	
	for(int ch=0; ch<2; ch++){ //for each channel

	    //calculate mean, variance, standard dev of the statistic over all chunks
	    for(int i=ch* numChunks/2; i<(ch+1) * numChunks/2; i++){
		gc->variance[csi][i] = variance(gc->sqMean[csi][i], gc->mean[csi][i]);
		tmean[ch] += statistic[csi][i]/(numChunks/2);
		tsqMean[ch]+=pow(statistic[csi][i], 2)/(numChunks/2);
	    }
	    tvar[ch] = variance(tsqMean[ch], tmean[ch]);
	    trms[ch] = sqrt(tvar[ch]);
	    
            //handle rfi
	    for(int i=ch* numChunks/2; i<(ch+1) * numChunks/2; i++){
		if(abs(statistic[csi][i] - tmean[ch]) > set->n_sigma_null* trms[ch]){
		 o[ch]++;
		 //mimic logical OR of flagged chunks in each channel
		 if(gc->isOutlier[csi][i%2] == 0){//other channel didn't flag this chunk
		 	gc->isOutlier[csi][i%2] = 1; //flag as outlier
			outliersOR++;
		 }
		
		 if(set->null_RFI) CHK(cudaMemsetAsync(&(gc->cfbuf[csi][i*gc->chunkSize]), 0, gc->chunkSize, cs)); //zero out outliers for FFT
		 
		 for(uint32_t j =0; j<gc->chunkSize; j++)
		     gc->outlierBuf[j] = buf[2*(i%2 * gc->chunkSize + j) + ch]; //deinterleave data in order to write out to file 
                 		 
		 //Write outlier to file
		 if(abs(statistic[csi][i] - tmean[ch]) > set->n_sigma_write* trms[ch])
		 	writerWriteRFI(wr, gc->outlierBuf, i%2 , ch);
	       }
	   }
	}
        clock_gettime(CLOCK_REALTIME, &now);
	//tprintfn("Time after finished writing: %f", (now.tv_sec-start.tv_sec)+(now.tv_nsec - start.tv_nsec)/1e9);


	//calculate approximate average of outliers per channel per sample (approximate because using wr->counter which might be a bit behind)
	int n = wr->counter; 
        for(int i=0; i <gc->nchan; i++)
	    gc->avgOutliersPerChannel[i]= (gc->avgOutliersPerChannel[i]*n + o[i])/(n+1);

	tprintfn(" ");
	tprintfn("RFI analysis: ");
	tprintfn("CH1 mean/var/rms: %f %f %f CH2 mean/var/rms: %f %f %f", tmean[0], tvar[0], trms[0], tmean[1], tvar[1], trms[1]);
	tprintfn("CH1 outliers: %d CH2 outliers: %d", o[0], o[1]); 
	tprintfn("CH1 average outliers: %f CH2 average outliers: %f", gc->avgOutliersPerChannel[0], gc->avgOutliersPerChannel[1]); 
    }
    cudaEventRecord(gc->eDoneRFI[csi],cs);

    //perform fft
    int status = cufftSetStream(gc->plan, cs);
    if(status !=CUFFT_SUCCESS) {
    	printf("CUFFTSETSTREAM failed\n");
 	exit(1);
    }
    status=cufftExecR2C(gc->plan, gc->cfbuf[csi], gc->cfft[csi]);
    cudaEventRecord(gc->eDoneFFT[csi], cs);
    if (status!=CUFFT_SUCCESS) {
      printf("CUFFT FAILED\n");
      exit(1);
    } 

    if (gc->nchan==1) {
      int psofs=0;
      for (int i=0; i<gc->ncuts; i++) {
	ps_reduce<<<gc->pssize[i], 1024, 0, cs>>> (gc->cfft[csi], &(gc->coutps[csi][psofs]), gc->ndxofs[i], gc->fftavg[i], 1);
	psofs+=gc->pssize[i];
      }
    } else if(gc->nchan==2){
	  // note we need to take into account the tricky N/2+1 FFT size while we do N/2 binning
	  // pssize+2 = transformsize+1
	  // note that pssize is the full *nchan pssize
      	  
	  //calculate power spectra corrections due to nulling out chunks flagged as RFI
	  float ch1Correction = numChunks/(numChunks - o[0]);  //correction for channel 1 power spectrum
	  float ch2Correction = numChunks/(numChunks - o[1]);  //correction for channel 2 power spectrum
	  float crossCorrection = numChunks/(numChunks - outliersOR); //correction for cross spectrum

	  int psofs=0;
	  for (int i=0; i<gc->ncuts; i++) {
	    ps_reduce<<<gc->pssize1[i], 1024, 0, cs>>> (&gc->cfft[csi][0], &(gc->coutps[csi][psofs]), gc->ndxofs[i], gc->fftavg[i], ch1Correction);
	    psofs+=gc->pssize1[i];
	    ps_reduce<<<gc->pssize1[i], 1024, 0, cs>>> (&gc->cfft[csi][(gc->fftsize/2+1)], &(gc->coutps[csi][psofs]), gc->ndxofs[i], gc->fftavg[i], ch2Correction);
	    psofs+=gc->pssize1[i];
	    ps_X_reduce<<<gc->pssize1[i], 1024, 0, cs>>> (&gc->cfft[csi][0], &gc->cfft[csi][(gc->fftsize/2+1)], 
					      &(gc->coutps[csi][psofs]), &(gc->coutps[csi][psofs+gc->pssize1[i]]),
					      gc->ndxofs[i], gc->fftavg[i], crossCorrection);
	    psofs+=2*gc->pssize1[i];
	}
  }
  else{
      printf("Can only handle 1 or 2 channels\n");
      exit(1);
  }
 
    CHK(cudaGetLastError());
    cudaEventRecord(gc->eDonePost[csi], cs);
    
  return true;
}
