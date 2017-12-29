/***********************************
***********************************
CUDA PART
***********************************
**********************************/

#define CUDA_COMPILE //to enable cuda types in gpucard.h
#include "gpucard.h"
#undef CUDA_COMPILE
#include "terminal.h"
#include "reduction.h"
#include "rfi.h"
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

//Print GPU properties
//inputs:
//       prop: pointer to structure containing device properties
//       dev: device number
void printDeviceProperties(cudaDeviceProp * prop, int dev){
    CHK(cudaGetDeviceProperties(prop, dev));
    printf("\nGPU properties \n====================\n");
    printf("Version number:                %d.%d\n",  prop->major, prop->minor);
    printf("Name:                          %s\n",  prop->name);
    printf("Total global memory:           %u\n",  prop->totalGlobalMem);
    printf("Total shared memory per block: %u\n",  prop->sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  prop->regsPerBlock);
    printf("Warp size:                     %d\n",  prop->warpSize);
    printf("Maximum memory pitch:          %u\n",  prop->memPitch);
    printf("Maximum threads per block:     %d\n",  prop->maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, prop->maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, prop->maxGridSize[i]);
    printf("Clock rate:                    %d\n",  prop->clockRate);
    printf("Total constant memory:         %u\n",  prop->totalConstMem);
    printf("Texture alignment:             %u\n",  prop->textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (prop->deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  prop->multiProcessorCount);
    printf("Kernel execution timeout:      %s\n\n",  (prop->kernelExecTimeoutEnabled ? "Yes" : "No"));
}

//Initialize instance of GPUCARD
//Input:
//      gc: instance of GPUCARD to initialize
//      set: settings
void gpuCardInit (GPUCARD *gc, SETTINGS *set) {
  
  //print out gpu device properties
  gc->devProp = (cudaDeviceProp *)malloc(sizeof(cudaDeviceProp));
  printDeviceProperties(gc->devProp, 0);  
  
  printf ("\n\nInitializing GPU\n");
  printf ("====================\n");
  printf ("Allocating GPU buffers\n");
  
  int nStreams=set->cuda_streams;
  gc->cbuf=(int8_t**)malloc(nStreams*sizeof(int8_t*));
  gc->cfbuf=(cufftReal**)malloc(nStreams*sizeof(cufftReal*));
  gc->cfft=(cufftComplex**)malloc(nStreams*sizeof(cufftComplex*));
  gc->coutps=(float**)malloc(nStreams*sizeof(float*));
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
  for (int i=0;i<nStreams;i++) {
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
//      rfi: structure containing rfi settings and memory for rfi statistics
//	set: settings
bool gpuProcessBuffer(GPUCARD *gc, int8_t **buf, WRITER *wr, RFI * rfi, SETTINGS *set) {
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
                 	float n=buf[0][i];
                	m1+=n; v1+=n*n;
                	n=buf[0][i+1];
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
                writerWritePS(wr,gc->outps, rfi->numOutliersNulled[gc->fstream]);
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
		;//exit(1);
    }

    gc->active_streams++;
    cudaStream_t cs= gc->streams[gc->bstream];
    cudaEventRecord(gc->eStart[csi], cs);
    
    //memory copy
    CHK(cudaMemcpyAsync(gc->cbuf[csi], buf[0], gc->bufsize , cudaMemcpyHostToDevice,cs));
    
    //floatize
    cudaEventRecord(gc->eDoneCopy[csi], cs);
    int threadsPerBlock = gc->threads;
    int blocksPerGrid = gc->bufsize / threadsPerBlock/FLOATIZE_X;
    if (gc->nchan==1) 
      floatize_1chan<<<blocksPerGrid, threadsPerBlock, 0, cs>>>(gc->cbuf[csi],gc->cfbuf[csi]);
    else 
      floatize_2chan<<<blocksPerGrid, threadsPerBlock, 0, cs>>>(gc->cbuf[csi],gc->cfbuf[csi],&(gc->cfbuf[csi][gc->fftsize]));
    cudaEventRecord(gc->eDoneFloatize[csi], cs);
    
    //RFI rejection 
    if(gc->nchan == 2 && rfi->statFlags != 0 && (rfi->nSigmaNull > 0 || rfi->nSigmaWrite > 0)){
  	collectRFIStatistics(rfi, gc, csi);
        nullRFI(rfi, gc, csi, wr);
   	writeRFI(rfi, gc, csi, wr, buf[0]);
    }
    cudaEventRecord(gc->eDoneRFI[csi],gc->streams[csi]);

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

    //compute spectra
    if (gc->nchan==1) {
      int psofs=0;
      for (int i=0; i<gc->ncuts; i++) {
	ps_reduce<<<gc->pssize[i], 1024, 0, cs>>> (gc->cfft[csi], &(gc->coutps[csi][psofs]), gc->ndxofs[i], gc->fftavg[i], 1);
	psofs+=gc->pssize[i];
      }
    } else if(gc->nchan==2){
	  // note we need to take into account the tricky N/2+1 FFT size while we do N/2 binning
	  // pssize+2 = transformsize+1
	  
	  //calculate power spectra corrections due to nulling out chunks flagged as RFI
	  int numChunks = gc->bufsize/rfi->chunkSize;
	  float ch1Correction = numChunks*1.0/(numChunks - rfi->numOutliersNulled[csi][0]);  //correction for channel 1 power spectrum
	  float ch2Correction = numChunks*1.0/(numChunks - rfi->numOutliersNulled[csi][1]);  //correction for channel 2 power spectrum
	  float crossCorrection = numChunks*1.0/(numChunks - rfi->outliersOR[csi]); //correction for cross spectrum
	  
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
