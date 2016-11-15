/***********************************
***********************************
CUDA PART
***********************************
**********************************/

#include "gpucard.h"

#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <stdio.h>
#include "math.h"

#define FLOATIZE_X 2


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

void gpuCardInit (GPUCARD *gc, SETTINGS *set) {
  printf ("\n\nInitializing GPU\n");
  printf ("====================\n");
  printf ("Allocating GPU buffers\n");
  int Nb=set->cuda_streams;
  gc->cbuf=(void**)malloc(Nb*sizeof(void*));
  gc->cfbuf=(void**)malloc(Nb*sizeof(void*));
  gc->cfft=(void**)malloc(Nb*sizeof(void*));
  gc->coutps=(void**)malloc(Nb*sizeof(void*));
  int nchan=gc->nchan=1+(set->channel_mask==3);
  if ((nchan==2) and (FLOATIZE_X%2==1)) {
    printf ("Need FLOATIZE_X even for two channels\n");
    exit(1);
  }
  gc->fftsize=set->fft_size;
  uint32_t bufsize=gc->bufsize=set->fft_size*nchan;
  uint32_t transform_size=(set->fft_size/2+1)*nchan;
  gc->fftavg=set->fft_avg;
  // first sort  reflections etc.
  //
  float nunyq=set->sample_rate/2;
  float numin, numax;
  numin=set->nu_min;
  numax=set->nu_max;
  printf ("%f %f\n",numax,nunyq);
  while (fabs(numin)>nunyq) numin-=set->sample_rate;
  while (fabs(numax)>nunyq) numax-=set->sample_rate;
  numin=abs(numin);
  numax=abs(numax);
  if (numax<numin) { float t=numin; numin=numax; numax=t; }
  printf ("Frequencies %f - %f Mhz appear as %f - %f \n",set->nu_min/1e6, set->nu_max/1e6,
	  numin/1e6, numax/1e6);
  float dnu=nunyq/(set->fft_size/2+1);
  int imin=int(numin/dnu);
  if (imin==0) imin=1;
  int imax=int(numax/dnu)+1;
  gc->pssize1=(imax-imin)/set->fft_avg;
  gc->ndxofs=imin;
  if ((imax-imin)%set->fft_avg>0) gc->pssize1+=1;
  imax=imin+gc->pssize1*set->fft_avg;
  numin=imin*dnu;
  numax=imax*dnu;
  set->nu_min=numin;
  set->nu_max=numax;
  printf ("Actual freq range: %f - %f MHz (edges!)\n",numin/1e6, numax/1e6);
  printf ("# PS offset, #PS bins: %i %i\n",gc->ndxofs,gc->pssize1);
  if (nchan==2)
    gc->pssize=gc->pssize1*4; // for other and two crosses
  else
    gc->pssize=gc->pssize1; // just one power spectrum
  for (int i=0;i<Nb;i++) {
    uint8_t** cbuf=(uint8_t**)&(gc->cbuf[i]);
    CHK(cudaMalloc(cbuf,bufsize));
    cufftReal** cfbuf=(cufftReal**)&(gc->cfbuf[i]);
    CHK(cudaMalloc(cfbuf, bufsize*sizeof(cufftReal)));
    cufftComplex** ffts=(cufftComplex**)&(gc->cfft[i]);
    CHK(cudaMalloc(ffts,transform_size*sizeof(cufftComplex)));
    float** coutps=(float**)&(gc->coutps[i]);
    CHK(cudaMalloc(coutps,gc->pssize*sizeof(float)));
  }
  CHK(cudaHostAlloc(&gc->outps, gc->pssize*sizeof(float), cudaHostAllocDefault));

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
  gc->nstreams=set->cuda_streams;
  gc->threads=set->cuda_threads;
  if (gc->nstreams<1) {
    printf ("Cannot relly work with less than one stream.\n");
    exit(1);
  }
  gc->streams=malloc(gc->nstreams*sizeof(cudaStream_t));
  gc->eStart=malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneCopy=malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneFloatize=malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneFFT=malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDonePost=malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneCopyBack=malloc(gc->nstreams*sizeof(cudaEvent_t));
  cudaEvent_t* eStart=(cudaEvent_t*)(gc->eStart);
  cudaEvent_t* eDoneCopy=(cudaEvent_t*)(gc->eDoneCopy);
  cudaEvent_t* eDoneFloatize=(cudaEvent_t*)(gc->eDoneFloatize);
  cudaEvent_t* eDoneFFT=(cudaEvent_t*)(gc->eDoneFFT);
  cudaEvent_t* eDonePost=(cudaEvent_t*)(gc->eDonePost);
  cudaEvent_t* eDoneCopyBack=(cudaEvent_t*)(gc->eDoneCopyBack);

  for (int i=0;i<gc->nstreams;i++) {
    CHK(cudaEventCreate(&eStart[i]));
    CHK(cudaEventCreate(&eDoneCopy[i]));
    CHK(cudaEventCreate(&eDoneFloatize[i]));
    CHK(cudaEventCreate(&eDoneFFT[i]));
    CHK(cudaEventCreate(&eDonePost[i]));
    CHK(cudaEventCreate(&eDoneCopyBack[i]));
  }
  gc->fstream=gc->bstream=gc->active_streams=0;
  printf ("GPU ready.\n");
}




/**
 * CUDA Kernel byte->float, 1 channel version
 *
 */
__global__ void floatize_1chan(int8_t* sample, cufftReal* fsample)  {
    int i = FLOATIZE_X*(blockDim.x * blockIdx.x + threadIdx.x);
    for (int j=0; j<FLOATIZE_X; j++) fsample[i+j]=float(sample[i+j]);
}

__global__ void floatize_2chan(int8_t* sample, cufftReal* fsample1, cufftReal* fsample2)  {
    int i = FLOATIZE_X*(blockDim.x * blockIdx.x + threadIdx.x);
    for (int j=0; j<FLOATIZE_X/2; j++) {
      fsample1[i/2+j]=float(sample[i+2*j]);
      fsample2[i/2+j]=float(sample[i+2*j+1]);
    }
}



/**
 * CUDA reduction sum
 * we will take bsize complex numbers starting at ffts[istart+bsize*blocknumber]
 * and their copies in NCHUNS, and add the squares
 **/


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
  


/** 
 * CROSS power spectrum reducer
 **/
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
  



void printDt (cudaEvent_t cstart, cudaEvent_t cstop) {
  float gpu_time;
  CHK(cudaEventElapsedTime(&gpu_time, cstart, cstop));
  printf (" %3.2fms ", gpu_time);
}

void printTiming(GPUCARD *gc, int i) {
  printf ("GPU timing (copy/floatize/fft/post/copyback): ");
  cudaEvent_t* eStart=(cudaEvent_t*)(gc->eStart);
  cudaEvent_t* eDoneCopy=(cudaEvent_t*)(gc->eDoneCopy);
  cudaEvent_t* eDoneFloatize=(cudaEvent_t*)(gc->eDoneFloatize);
  cudaEvent_t* eDoneFFT=(cudaEvent_t*)(gc->eDoneFFT);
  cudaEvent_t* eDonePost=(cudaEvent_t*)(gc->eDonePost);
  cudaEvent_t* eDoneCopyBack=(cudaEvent_t*)(gc->eDoneCopyBack);
  printDt (eStart[i], eDoneCopy[i]);
  printDt (eDoneCopy[i], eDoneFloatize[i]);
  printDt (eDoneFloatize[i], eDoneFFT[i]);
  printDt (eDoneFFT[i], eDonePost[i]);
  printDt (eDonePost[i], eDoneCopyBack[i]);
  printf ("  \n");
}

bool gpuProcessBuffer(GPUCARD *gc, int8_t *buf, WRITER *wr) {
  // pointers and vars
  int8_t** cbuf=(int8_t**)(gc->cbuf);
  cufftReal** cfbuf=(cufftReal**)(gc->cfbuf);
  cufftComplex** cfft=(cufftComplex**)(gc->cfft);
  float** coutps=(float**)(gc->coutps);

  cudaEvent_t* eStart=(cudaEvent_t*)(gc->eStart);
  cudaEvent_t* eDoneCopy=(cudaEvent_t*)(gc->eDoneCopy);
  cudaEvent_t* eDoneFloatize=(cudaEvent_t*)(gc->eDoneFloatize);
  cudaEvent_t* eDoneFFT=(cudaEvent_t*)(gc->eDoneFFT);
  cudaEvent_t* eDonePost=(cudaEvent_t*)(gc->eDonePost);
  cudaEvent_t* eDoneCopyBack=(cudaEvent_t*)(gc->eDoneCopyBack);
  cudaStream_t* streams=(cudaStream_t*)gc->streams;

  if (gc->nstreams==1) {
    /// non-streamed version
    cudaEventRecord(eStart[0], 0);
    CHK(cudaMemcpy(cbuf[0], buf, gc->bufsize , cudaMemcpyHostToDevice));
    cudaEventRecord(eDoneCopy[0], 0);
    int threadsPerBlock = gc->threads;
    int blocksPerGrid = gc->bufsize / threadsPerBlock/FLOATIZE_X;
    if (gc->nchan==1) 
      floatize_1chan<<<blocksPerGrid, threadsPerBlock >>>(cbuf[0],cfbuf[0]);
    else 
      floatize_2chan<<<blocksPerGrid, threadsPerBlock >>>(cbuf[0],cfbuf[0],&(cfbuf[0][gc->fftsize]));
    CHK(cudaGetLastError());

    cudaEventRecord(eDoneFloatize[0], 0);
    int status=cufftExecR2C(gc->plan, cfbuf[0], cfft[0]);
    cudaEventRecord(eDoneFFT[0], 0);
    if (status!=CUFFT_SUCCESS) {
      printf("CUFFT FAILED\n");
      exit(1);
    }    

    if (gc->nchan==1)
      ps_reduce<<<gc->pssize, 1024>>> (&cfft[0][0], coutps[0], gc->ndxofs, gc->fftavg);
    else {
      // note we need to take into account the tricky N/2+1 FFT size while we do N/2 binning
      // pssize+2 = transformsize+1
      // note that pssize is the full *nchan pssize
      ps_reduce<<<gc->pssize1, 1024>>> (&cfft[0][0], coutps[0], gc->ndxofs, gc->fftavg);
      ps_reduce<<<gc->pssize1, 1024>>> (&cfft[0][(gc->fftsize/2+1)], 
                                         &(coutps[0][gc->pssize1]), gc->ndxofs, gc->fftavg);
      ps_X_reduce<<<gc->pssize1, 1024>>> (&cfft[0][0], &cfft[0][(gc->fftsize/2+1)], 
					  &(coutps[0][2*gc->pssize1]), &(coutps[0][3*gc->pssize1]),
					  gc->ndxofs, gc->fftavg);
    }
    CHK(cudaGetLastError());
    cudaEventRecord(eDonePost[0], 0);
    CHK(cudaMemcpy(gc->outps,coutps[0], gc->pssize*sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(eDoneCopyBack[0], 0);
    cudaThreadSynchronize();
    printTiming(gc,0);
    // now find some statistic over subsamples of samples
    uint32_t bs=gc->bufsize;
    uint32_t step=gc->bufsize/(32768);
    float fac=bs/step;
    float m1=0.,m2=0.,v1=0.,v2=0.;
    for (int i=0; i<bs; i+=step) {
      float n=buf[i];
      m1+=n; v1+=n*n;
      n=buf[i+1];
      m2+=n; v2+=n*n;
    }
    m1/=fac; v1=sqrt(v1/fac-m1*m1);
    m2/=fac; v2=sqrt(v2/fac-m1*m1);
    printf ("CH1 min/rms: %f %f   CH2 min/rms: %f %f   \n",m1,v1,m2,v2);
    writerWritePS(wr,gc->outps);
 } else {
    // streamed version
    printf ("Streamed version not ready.\n");
    exit(1);
    
    // first check if there are buffers to store
    while (gc->active_streams>0) {
      // process done streams
      // IMPLEMENT
    }
    // add a new stream
    gc->active_streams++;
    int csi=gc->bstream = (++gc->bstream)%(gc->nstreams);
    cudaStream_t cs= streams[gc->bstream];
    cudaEventRecord(eStart[csi], cs);
    CHK(cudaMemcpyAsync(cbuf[csi], buf, gc->bufsize , cudaMemcpyHostToDevice,cs));
    cudaEventRecord(eDoneCopy[csi], cs);
    int threadsPerBlock = gc->threads;
    int blocksPerGrid = gc->bufsize / threadsPerBlock/FLOATIZE_X;
    if (gc->nchan==1) 
      floatize_1chan<<<blocksPerGrid, threadsPerBlock, 0, cs>>>(cbuf[csi],cfbuf[csi]);
    else 
      floatize_2chan<<<blocksPerGrid, threadsPerBlock, 0, cs>>>(cbuf[csi],cfbuf[csi],&(cfbuf[csi][gc->fftsize]));
    cudaEventRecord(eDoneFloatize[csi], cs);
    int status=cufftExecR2C(gc->plan, cfbuf[csi], cfft[csi]);
  
    cudaEventRecord(eDoneFFT[csi], cs);
    cudaEventRecord(eDonePost[csi], cs);
  }



  
  return true;
}
