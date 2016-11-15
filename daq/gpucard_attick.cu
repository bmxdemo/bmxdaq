/***********************************
***********************************
CUDA PART
***********************************
**********************************/





/* Basic parameters of ADC */
#define SAMPLING_RATE 2.7e9
#define DELTA_T (1.0/SAMPLING_RATE)

/* SIZE OF FFT, BUFFERS 
we aim at 1ms = 2.7MSamples, 
2**21 is 0.8 ms 
2**23 is 0.99ms
*/
#define BUFFER_SIZE (268435456)
#define NUM_FFT 1
#define FFT_SIZE (BUFFER_SIZE/NUM_FFT)
#define TRANSFORM_SIZE (FFT_SIZE/2+1)

/* Size of a chunk we process in one go
At 2.7 MS, this 99.4 ms*/

#define DELTA_NU (1./(DELTA_T*FFT_SIZE))

/* numin and numax in freq in MHZ */
#define NUMIN 600.0
#define NUMAX 1300.0
/* We average over this many FFT bins */
#define NUAVG (16384/NUM_FFT)
/* and we get this many bins */


#define FLOATIZE_X 8
#define THREADS_DEFAULT 1024

void print_settings();
inline int num_nubins() { return (floor((NUMAX-NUMIN)*1e6/DELTA_NU/NUAVG)+1); }



#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <assert.h>

#include "gpucard.h"


#include <stdio.h>
#include "math.h"

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define CHK( err ) (HandleError( err, __FILE__, __LINE__ ))


uint8_t* alloc_sample_buffer() {
  uint8_t *p;
  /* size of uint8_t is one, explicityly */
  CHK(cudaHostAlloc(&p, BUFFER_SIZE, cudaHostAllocDefault));
 return p;
}

float* alloc_power() {
  float *p;
  /* size of uint8_t is one, explicityly */
  CHK(cudaHostAlloc(&p, num_nubins()*sizeof(float), cudaHostAllocDefault));
  return p;
}


void print_timing (cudaEvent_t* start, cudaEvent_t* stop, const char* what) {
  float gpu_time;
  CHK(cudaEventElapsedTime(&gpu_time, *start, *stop));
  printf ("Timing %s : %fms \n",what, gpu_time);
}


/**
 * CUDA Kernel byte->float
 *
 */
__global__ void floatize(uint8_t* sample,cufftReal* fsample)  {
    int i = FLOATIZE_X*(blockDim.x * blockIdx.x + threadIdx.x);
    for (int j=0; j<FLOATIZE_X; j++) fsample[i+j]=float(sample[i+j]-128);
}

/**
 * CUDA reduction sum
 * we will take bsize complex numbers starting at ffts[istart+bsize*blocknumber]
 * and their copies in NCHUNS, and add the squares
 **/


__global__ void ps_reduce(cufftComplex *ffts, float* output_ps, size_t istart) {
  int tid=threadIdx.x;
  int bl=blockIdx.x;
  int nth=blockDim.x;
  __shared__ float work[1024];
//  assert (tid<NUAVG);
  //global pos
  size_t pos=istart+bl*NUAVG+tid;
  //chunk pos
  size_t cpos=tid;
  work[tid]=0;
  size_t chunk=0;
  while (chunk<NUM_FFT) {
//    assert (pos<NUM_FFT*TRANSFORM_SIZE);
    work[tid]+=ffts[pos].x*ffts[pos].x+ffts[pos].y*ffts[pos].y;
    if (cpos+nth<NUAVG) {
      cpos+=nth;
      pos+=nth;
    } else {
      chunk++;
      pos=chunk*TRANSFORM_SIZE+istart+bl*NUAVG+tid;
      cpos=tid;
    }
  }

  // now do the three reduce.
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
  


void cuda_test(uint8_t *buf, float* freq, float*power) {

  // cuda buffer and float buffer
  uint8_t *cbuf;
  CHK(cudaMalloc(&cbuf,BUFFER_SIZE));
  cufftReal *cfbuf;
  CHK(cudaMalloc(&cfbuf,BUFFER_SIZE*sizeof(cufftReal)));
  cufftComplex *ffts;
  CHK(cudaMalloc(&ffts,TRANSFORM_SIZE*NUM_FFT*sizeof(cufftComplex)));
  int istart=int(NUMIN*1e6/DELTA_NU)-NUAVG/2;
  for (size_t i=0;i<num_nubins();i++) freq[i]=(istart+i*NUAVG)*DELTA_NU/1e6;
  // device power
  float *cpower;
  CHK(cudaMalloc(&cpower,num_nubins()*sizeof(float)));
  
  
  cufftHandle plan;
  //int oembed=TRANSFORM_SIZE*NUM_FFT+1;
  int fftsize=FFT_SIZE;
  int status=cufftPlanMany(&plan, 1, &fftsize, NULL, 0, 0, 
        NULL, TRANSFORM_SIZE,1, CUFFT_R2C, NUM_FFT);
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

  cudaEvent_t tstart, tcpy,tfloatize,tfft,treduce,tcopyback;
  CHK(cudaEventCreate(&tstart));
  CHK(cudaEventCreate(&tcpy));
  CHK(cudaEventCreate(&tfloatize));
  CHK(cudaEventCreate(&tfft));
  CHK(cudaEventCreate(&treduce));
  CHK(cudaEventCreate(&tcopyback));

  cudaEventRecord(tstart, 0);
  // copy to device
  CHK(cudaMemcpy(cbuf,buf, BUFFER_SIZE, cudaMemcpyHostToDevice));

  cudaEventRecord(tcpy, 0);
  
  // floatize
  int threadsPerBlock = THREADS_DEFAULT;
  int blocksPerGrid = BUFFER_SIZE / threadsPerBlock/FLOATIZE_X;
  floatize<<<blocksPerGrid, threadsPerBlock>>>(cbuf,cfbuf);
  cudaEventRecord(tfloatize, 0);
  CHK(cudaGetLastError());
  
  status=cufftExecR2C(plan, cfbuf, ffts);
  cudaEventRecord(tfft, 0);
  if (status!=CUFFT_SUCCESS) {
     printf("CUFFT FAILED\n");
     exit(1);
  }    


  // now launch the final kernel
  while (threadsPerBlock>NUAVG) threadsPerBlock/=2;
  // ps_reduce<<<num_nubins(),threadsPerBlock>>>(ffts,cpower, istart);
  ps_reduce<<<num_nubins(), threadsPerBlock>>>(ffts,cpower, istart);
  cudaEventRecord(treduce, 0);
  // copy results over
  CHK(cudaMemcpy(power,cpower, num_nubins()*sizeof(float), cudaMemcpyDeviceToHost));
  cudaEventRecord(tcopyback, 0);

  cudaThreadSynchronize();
  print_timing(&tstart,&tcpy,"MEM CPY");
  print_timing(&tcpy,&tfloatize,"FLOATIZE");
  print_timing(&tfloatize,&tfft,"FFT");
  print_timing(&tfft,&treduce,"REDUCE");
  print_timing(&tfft,&treduce,"COPYBACK");

#ifdef DEBUGREDUCE
  cufftComplex *hffts;
  CHK(cudaMallocHost(&hffts,TRANSFORM_SIZE*NUM_FFT*sizeof(cufftComplex)));
  CHK(cudaMemcpy(hffts,ffts, TRANSFORM_SIZE*NUM_FFT*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
  //now check first and last elements of the transform which should be real
  for (size_t i=0;i<NUM_FFT;i++) {
    cufftComplex f=hffts[i*TRANSFORM_SIZE];
    cufftComplex s=hffts[i*TRANSFORM_SIZE+1];
    cufftComplex ml=hffts[(i+1)*TRANSFORM_SIZE-2];
    cufftComplex l=hffts[(i+1)*TRANSFORM_SIZE-1];
    printf ("%i first %f %f , second %f %f, lastbyone %f %f, last %f %f \n",
	    (int)i, f.x,f.y,s.x,s.y,ml.x,ml.y, l.x,l.y);
  }
  //now do the powers
  for (size_t i=0;i<num_nubins();i++) {
    float pow=0;
    for (size_t j=0;j<NUM_FFT;j++) {
      for (size_t k=0;k<NUAVG;k++) {
        int pos=j*TRANSFORM_SIZE+istart+i*NUAVG+k;
	if (!(pos<TRANSFORM_SIZE*NUM_FFT)) {
	  printf ("SHIT %i %i %i %i\n",NUM_FFT, TRANSFORM_SIZE, NUAVG, num_nubins());
	  exit(1);
	}
	pow+=hffts[pos].x*hffts[pos].x+hffts[pos].y*hffts[pos].y;
      }
    }
    printf ("power %i %fMHz %f %f\n", (int)i,freq[i], pow, power[i]);
  }

#endif
}


void ztest() {
  uint8_t *cbuf;
  CHK(cudaMalloc(&cbuf,BUFFER_SIZE));
  // floatize
  int threadsPerBlock = 1024;
  int blocksPerGrid = 32768;
  int Nth=
  printf ("%i %i",threadsPerBlock, blocksPerGrid);
  //  floatize<<<blocksPerGrid, threadsPerBlock>>>(cbuf);
  CHK(cudaGetLastError());

}
