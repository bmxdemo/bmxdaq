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
  printf("Total global memory:           %lu\n",  prop->totalGlobalMem);
  printf("Total shared memory per block: %lu\n",  prop->sharedMemPerBlock);
  printf("Total registers per block:     %d\n",  prop->regsPerBlock);
  printf("Warp size:                     %d\n",  prop->warpSize);
  printf("Maximum memory pitch:          %lu\n",  prop->memPitch);
  printf("Maximum threads per block:     %d\n",  prop->maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
  printf("Maximum dimension %d of block:  %d\n", i, prop->maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
  printf("Maximum dimension %d of grid:   %d\n", i, prop->maxGridSize[i]);
  printf("Clock rate:                    %d\n",  prop->clockRate);
  printf("Total constant memory:         %lu\n",  prop->totalConstMem);
  printf("Texture alignment:             %lu\n",  prop->textureAlignment);
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
  gc->calibrating=false;
  gc->calibrated=false;
  gc->calibok=0;
  int nchan=gc->nchan=1+(set->channel_mask==3);
  if ((nchan==2) and (FLOATIZE_X%2==1)) {
    printf ("Need FLOATIZE_X even for two channels\n");
    exit(1);
  }

  if(!(OPS_PER_THREAD>0) ||  !((OPS_PER_THREAD & (OPS_PER_THREAD-1)) == 0)){
    printf("Need OPS_PER_THREAD to be a power of 2.\n");
    exit(1);
  }

  printf ("\n\nInitializing GPU\n");
  printf ("====================\n");
  printf ("Allocating GPU buffers\n");
  
  int nStreams=set->cuda_streams;
  int nCards=(set->card_mask==3) + 1;
  gc->fftsize=set->fft_size;
  uint32_t bufsize=gc->bufsize=set->fft_size*nchan;
  uint32_t transform_size=gc->transform_size=(set->fft_size/2+1);
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
    if (nchan==2){ 
      if (nCards==1)
	gc->pssize[i]=gc->pssize1[i]*4; // for two channels and two crosses
	else
	gc->pssize[i]=gc->pssize1[i]*16; // for 4 channels and 6*2 crosses
    } else {
      gc->pssize[i]=gc->pssize1[i]*nCards; // just one power spectrum
    }

    gc->tot_pssize+=gc->pssize[i];
    printf ("   Actual freq range: %f - %f MHz (edges!)\n",numin/1e6, numax/1e6);
    printf ("   # PS offset, #PS bins: %i %i\n",gc->ndxofs[i],gc->pssize1[i]);
  }
  CHK(cudaHostAlloc(&gc->outps, gc->tot_pssize*sizeof(float), cudaHostAllocDefault));

  //allocating GPU buffers
  gc->cbuf=(int8_t***)malloc(nStreams*sizeof(int8_t**));
  gc->cfbuf=(cufftReal**)malloc(nStreams*sizeof(cufftReal*));
  gc->cfft=(cufftComplex**)malloc(nStreams*sizeof(cufftComplex*));
  gc->coutps=(float**)malloc(nStreams*sizeof(float*));
  for (int i=0;i<nStreams;i++) {
    gc->cbuf[i]=(int8_t**)malloc(nCards*sizeof(int8_t*));
    for(int j=0; j<nCards; j++)
      CHK(cudaMalloc(&(gc->cbuf[i][j]),bufsize));
    CHK(cudaMalloc(&gc->cfbuf[i], bufsize*nCards*sizeof(cufftReal)));
    CHK(cudaMalloc(&gc->cfft[i],transform_size*nchan*nCards*sizeof(cufftComplex)));
    CHK(cudaMalloc(&gc->coutps[i],gc->tot_pssize*sizeof(float)));
    CHK(cudaMalloc(&gc->cmeasured_delay,nStreams*sizeof(int)));
  }

  printf ("Setting up CUFFT\n");
  //  int status=cufftPlanMany(&gc->plan, 1, (int*)&(set->fft_size), NULL, 0, 0, 
  //    NULL, 2*transform_size,1, CUFFT_R2C, nchan);
  int status=cufftPlan1d(&gc->plan, set->fft_size, CUFFT_R2C, nchan);

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

  status=cufftPlan1d(&gc->iplan, set->fft_size, CUFFT_C2R, 1); // inverse transform always for one channel only

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
  gc->eDoneCalib=(cudaEvent_t*)malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneStream=(cudaEvent_t*)malloc(gc->nstreams*sizeof(cudaEvent_t));
  for (int i=0;i<gc->nstreams;i++) {
    //create stream
    CHK(cudaStreamCreate(&gc->streams[i]));
    //create events for stream
    CHK(cudaEventCreate(&gc->eStart[i]));
    CHK(cudaEventCreate(&gc->eDoneCopy[i]));
    CHK(cudaEventCreate(&gc->eDoneFloatize[i]));
    CHK(cudaEventCreate(&gc->eDoneFFT[i]));
    CHK(cudaEventCreate(&gc->eDonePost[i]));
    CHK(cudaEventCreate(&gc->eDoneCalib[i]));
    CHK(cudaEventCreate(&gc->eDoneStream[i]));
  }
 
  gc->fstream = 0; //oldest running stream
  gc->bstream = -1; //newest stream (will become 0 when we actually start with first real stream)
  gc->active_streams = 0; //number of streams currently running
  
  printf ("GPU ready.\n");
}

//Convert bytes to floats, 1 channel version
//Inputs:
//   sample: array of bytes
//       fsample: array of floats to put output in
__global__ void floatize_1chan(int8_t* sample, cufftReal* fsample)  {
  int i = FLOATIZE_X*(blockDim.x * blockIdx.x + threadIdx.x);
  for (int j=0; j<FLOATIZE_X; j++) fsample[i+j]=float(sample[i+j]);
}


//Convert bytes to floats, 2 channel version
//Inputs:
//   sample: array of bytes with the 2 channels interleaved
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
void printDt (cudaEvent_t cstart, cudaEvent_t cstop, float * total, TWRITER * t) {
  float gpu_time;
  CHK(cudaEventElapsedTime(&gpu_time, cstart, cstop));
  tprintfn (t, 0, " %3.2fms ", gpu_time);
  *total +=gpu_time;
}

void printTiming(GPUCARD *gc, int i, TWRITER * t) {
  float totalTime = 0;
  tprintfn (t, 0, "GPU timing (copy/floatize/fft/post/calib): ");
  printDt (gc->eStart[i], gc->eDoneCopy[i], &totalTime, t);
  totalTime=0;
  printDt (gc->eDoneCopy[i], gc->eDoneFloatize[i], &totalTime, t);
  printDt (gc->eDoneFloatize[i], gc->eDoneFFT[i], &totalTime, t);
  printDt (gc->eDoneFFT[i], gc->eDonePost[i], &totalTime, t);
  printDt (gc->eDonePost[i], gc->eDoneCalib[i], &totalTime, t);
  tprintfn (t,1,"");
  tprintfn (t, 1, "GPU timing cumpute total: %3.2f ", totalTime);
}


void printLiveStat(SETTINGS *set, GPUCARD *gc, int8_t **buf, TWRITER *twr) {
  int nCards=(set->card_mask==3) + 1;

  if (set->print_meanvar) {
    // now find some statistic over subsamples of samples
    uint32_t bs=gc->bufsize;
    uint32_t step=gc->bufsize/(32768);
    float NSub=bs/step; // number of subsamples to take
    float m1=0.,m2=0.,v1=0.,v2=0.;
    float m3=0.,m4=0.,v3=0.,v4=0.;
    for (int i=0; i<bs; i+=step) { // take them in steps of step
      float n=buf[0][i];
      m1+=n; v1+=n*n;
      n=buf[0][i+1];
      m2+=n; v2+=n*n;
      if (nCards==2) {
	n=buf[1][i];        
	m3+=n; v3+=n*n;
	n=buf[1][i+1];
	m4+=n; v4+=n*n;

      }
    }
    m1/=NSub; v1=sqrt(v1/NSub-m1*m1); //mean and variance
    m2/=NSub; v2=sqrt(v2/NSub-m2*m2);
    tprintfn (twr,1,"CH1 mean/rms: %f %f   CH2 mean/rms: %f %f   ",m1,v1,m2,v2);
    if (nCards==2) {
      m3/=NSub; v3=sqrt(v3/NSub-m3*m3); //mean and variance
      m4/=NSub; v4=sqrt(v4/NSub-m4*m4);
      tprintfn (twr,1,"CH3 mean/rms: %f %f   CH4 mean/rms: %f %f   ",m3,v3,m4,v4);
    }
  }
  if (set->check_CH2) {
    // heuristic to see if CH2 is OK.
    float mean_card1=0;
    float mean_card2=0;
    int count=0;
    float numin=set->nu_min[0];
    float nustep=(set->nu_max[0]-set->nu_min[0])/(gc->pssize1[0]);
    int ofs2=gc->pssize1[0];
    int ofs4=3*gc->pssize1[0];

    for (int j=0; j<gc->pssize1[0];j++) { // check just cut 0
      float f=numin+nustep*j;
      if ((f>1560e6-1100e6) && (f<1640e6-1100e6)) {
	count++;
	mean_card1+=gc->outps[ofs2+j];
	if (nCards==2) mean_card2+=gc->outps[ofs4+j];
      }
    }
    if (count>0) {
      mean_card1/=(count*1e11)*(set->fft_avg[0]/8192);
      int ok=0;
      if (nCards==1) {
	ok=mean_card1<1;
	tprintfn (twr,0,"CH2 check : %f : ",mean_card1);

      } else {
	mean_card2/=(count*1e11)*(set->fft_avg[0]/8192);;
	ok=((mean_card1<1) && (mean_card2<1));
	tprintfn (twr,0,"CH2 check : %f / %f : ",mean_card1, mean_card2);
      }
      if (ok) tprintfn(twr,1, " OK "); else
		tprintfn(twr,1, " NOT OK !!!");
    }
  }

  if (set->print_maxp) {
    // find max power in each cutout in each channel.
    int of1=0; // CH1 auto

    for (int i=0; i<gc->ncuts; i++) {
      int of2=of1+gc->pssize1[i]; //CH2 auto 
      int of3=of1+2*gc->pssize1[i]; // CH3 auto
      int of4=of1+3*gc->pssize1[i]; // CH4 auto

      float ch1p=0, ch2p=0, ch3p=0, ch4p=0;
      int ch1i=0, ch2i=0, ch3i=0, ch4i=0;

      for (int j=0; j<gc->pssize1[i];j++) {
	if (gc->outps[of1+j] > ch1p) {ch1p=gc->outps[of1+j]; ch1i=j;}
	if (gc->outps[of2+j] > ch2p) {ch2p=gc->outps[of2+j]; ch2i=j;}
	if (nCards==2) {
	  if (gc->outps[of3+j] > ch3p) {ch3p=gc->outps[of3+j]; ch3i=j;}
	  if (gc->outps[of4+j] > ch4p) {ch4p=gc->outps[of4+j]; ch4i=j;}
	}
      }
      of1+=gc->pssize[i];  // next cutout 
      float numin=set->nu_min[i];
      float nustep=(set->nu_max[i]-set->nu_min[i])/(gc->pssize1[i]);
      float ch1f=(numin+nustep*(0.5+ch1i))/1e6;
      float ch2f=(numin+nustep*(0.5+ch2i))/1e6;
      tprintfn (twr,1,"Peak pow (cutout %i): CH1 %f at %f MHz;   CH2 %f at %f MHz  ",
		i,log(ch1p),ch1f,log(ch2p),ch2f);
      if (nCards==2) {
	float ch3f=(numin+nustep*(0.5+ch3i))/1e6;
	float ch4f=(numin+nustep*(0.5+ch4i))/1e6;
	tprintfn (twr,1,"Peak pow (cutout %i): CH3 %f at %f MHz;   CH4 %f at %f MHz  ",
		  i,log(ch3p),ch3f,log(ch4p),ch4f);
      }
    }
  }

  if (set->measure_delay || gc->calibrating) {
    float delayms=float(gc->last_measured_delay)*1.0/(set->sample_rate)*1e3;
    tprintfn (twr,1, "Calibrating: %i/%i Last measured delay: %i samples = %f ms. ",
	      gc->ndelays, NUM_DELAYS, gc->last_measured_delay, delayms);
  } else {
    if  (gc->calibrated) 
      tprintfn (twr,1, "DCal: OK: %i/%i  Val %f +- %fms Applied delay: %iB+%iS %iB+%iS",
		gc->calibok, NUM_DELAYS, gc->calibmean_ms, gc->calibrms_ms, 
		set->bufdelay[0], set->delay[0], 
		set->bufdelay[1], set->delay[1]);
    else
      tprintfn (twr,1, "DCal: Failed: %i/%i  Applied delay: %iB+%iS %iB+%iS ",
		gc->calibok, NUM_DELAYS, set->bufdelay[0], set->delay[0], 
		set->bufdelay[1], set->delay[1]);
  }
}

//process calibration data and  stops calibrationc process
void processCalibration(GPUCARD *gc, SETTINGS *set) {
  gc->calibrating=false;
  long int mean=0;
  long int var=0;
  int numok=0;
  const int OK=1500000;  // 1.4 ms
  for (int i=0; i<NUM_DELAYS; i++) {
    if (abs(gc->delays[i])<OK) {
      mean+=gc->delays[i];
      var+=gc->delays[i]*gc->delays[i];
      numok++;
    }  
  }
  gc->calibok=numok;
  if (numok>NUM_DELAYS/2) {
    gc->calibrated=true;
    mean/=numok;
    var/=numok;
    gc->calibmean=mean;
    if (gc->calibmean>0) {
      set->delay[0]+=gc->calibmean;
    } else if (gc->calibmean<0) {
      set->delay[1]+= (-gc->calibmean);
    }
    int mindel=std::min(set->delay[0],set->delay[1]);
    set->delay[0]-=mindel;
    set->delay[1]-=mindel;
    gc->calibrms = int(sqrt(var-mean*mean));
    gc->calibmean_ms= gc->calibmean*1.0/(set->sample_rate)*1e3;
    gc->calibrms_ms= gc->calibrms*1.0/(set->sample_rate)*1e3;

  } else
    gc->calibrated=false;
}

void startCalib(GPUCARD *gc) {
  gc->calibrating=true;
  gc->ndelays=0;
}


//Process one data packet from the digitizer
//Input:
//  gc: graphics card
//      buf: data from digitizer
//      pbuf: old data from digitizer (to implement delay) 
//      wr: writer to write out power spectra and outliers to files

//  set: settings
int gpuProcessBuffer(GPUCARD *gc, int8_t **buf_one, int8_t **buf_two, WRITER *wr, TWRITER *twr, SETTINGS *set) {
  //streamed version
  //Check if other streams are finished and proccess the finished ones in order (i.e. print output to file)

  CHK(cudaGetLastError());

  int nCards=(set->card_mask==3) + 1;

  int8_t* buf[2];
  int8_t* pbuf[2];
  buf[0]=buf_one[set->bufdelay[0]];
  pbuf[0]=buf_one[set->bufdelay[0]+1];
  buf[1]=buf_two[set->bufdelay[1]];
  pbuf[1]=buf_two[set->bufdelay[1]+1];

  while(gc->active_streams > 0){
    // printf ("S:%i ", cudaEventQuery(gc->eStart[gc->fstream])==cudaSuccess);
    // printf ("%i ", cudaEventQuery(gc->eDoneCopy[gc->fstream])==cudaSuccess);
    // printf ("%i ", cudaEventQuery(gc->eDoneFloatize[gc->fstream])==cudaSuccess);
    // printf ("%i ", cudaEventQuery(gc->eDoneFFT[gc->fstream])==cudaSuccess);
    // printf ("%i [%i]\n ", cudaEventQuery(gc->eDonePost[gc->fstream])==cudaSuccess, gc->fstream);
    if(cudaEventQuery(gc->eDoneStream[gc->fstream])==cudaSuccess){
      int fstream=gc->fstream;
      if (!gc->calib[fstream]) {
	cudaMemcpy(gc->outps,gc->coutps[fstream], 
		   gc->tot_pssize*sizeof(float), cudaMemcpyDeviceToHost);
      } else {
	cudaMemcpy(&gc->last_measured_delay,&(gc->cmeasured_delay[fstream]),
		   sizeof(int), cudaMemcpyDeviceToHost);
	gc->delays[gc->ndelays]=gc->last_measured_delay;
	gc->ndelays++;
	if (gc->ndelays==NUM_DELAYS) processCalibration(gc,set);
      }

      if (gc->active_streams==1) {
	printTiming(gc,fstream,twr);
	printLiveStat(set,gc,buf,twr);
	writerAccumulatePS(wr,gc->outps, twr,set);
      } else
	writerAccumulatePS(wr,gc->outps, NULL,set); // accumulate, but without talking
      gc->fstream = (++gc->fstream)%(gc->nstreams);
      gc->active_streams--;
    }
    else 
      break;      
  }  

  if(gc->active_streams == gc->nstreams){ //if no empty streams
       	return false;
  }

  gc->active_streams++;
  int csi = gc->bstream = (++gc->bstream)%(gc->nstreams); //add new stream

  cudaStream_t cs= gc->streams[gc->bstream];
  cudaEventRecord(gc->eStart[csi], cs);
  
  //memory copy
  for(int i=0; i<nCards; i++) {
    if (set->delay[i]==0) 
      cudaMemcpyAsync(gc->cbuf[csi][i], buf[i], gc->bufsize , cudaMemcpyHostToDevice,cs);
    else {
      if (set->delay[i]>gc->fftsize) {printf ("Pathological delay.\n"); exit(1);}
      unsigned ofs=set->delay[i]*gc->nchan;
      cudaMemcpyAsync(&gc->cbuf[csi][i][ofs], buf[i], gc->bufsize-ofs , cudaMemcpyHostToDevice,cs);
      if (pbuf[i])
      	cudaMemcpyAsync(gc->cbuf[csi][i], &pbuf[i][gc->bufsize-ofs], ofs , cudaMemcpyHostToDevice,cs);
    }
  }
  //floatize
  cudaEventRecord(gc->eDoneCopy[csi], cs);
  int threadsPerBlock = gc->threads;
  int blocksPerGrid = gc->bufsize / threadsPerBlock/FLOATIZE_X;
  if (gc->nchan==1) 
    floatize_1chan<<<blocksPerGrid, threadsPerBlock, 0, cs>>>(gc->cbuf[csi][0],gc->cfbuf[csi]);
  else
    for(int i=0; i<nCards; i++)
      floatize_2chan<<<blocksPerGrid, threadsPerBlock, 0, cs>>>
        (gc->cbuf[csi][i],&(gc->cfbuf[csi][gc->fftsize*2*i]),&(gc->cfbuf[csi][gc->fftsize*(2*i+1)]));
  cudaEventRecord(gc->eDoneFloatize[csi], cs);
  
  //perform fft
  int status = cufftSetStream(gc->plan, cs);
  if(status !=CUFFT_SUCCESS) {
    printf("CUFFSTETSTREAM failed\n");
    exit(1);
  }
  for(int i=0; i<nCards;i++){
    status=cufftExecR2C(gc->plan, &(gc->cfbuf[csi][gc->bufsize*i]), &(gc->cfft[csi][2*i*gc->transform_size]));
    if (status!=CUFFT_SUCCESS) {
      printf("CUFFT FAILED\n");
      exit(1);
    } 
  } 
  cudaEventRecord(gc->eDoneFFT[csi], cs);
  
  if (!set->measure_delay & !gc->calibrating) {
    gc->calib[csi]=false;
    //compute spectra
    if (gc->nchan==1) {
      int psofs=0;
      for (int i=0; i<gc->ncuts; i++) {
	ps_reduce<<<gc->pssize[i], 1024, 0, cs>>> (gc->cfft[csi], &(gc->coutps[csi][psofs]), 
						   gc->ndxofs[i], gc->fftavg[i]);
	psofs+=gc->pssize[i];
      }
    } 
    else if(gc->nchan==2){
      // note we need to take into account the tricky N/2+1 FFT size while we do N/2 binning
      // pssize+2 = transformsize+1
        
      int psofs=0;
      for (int i=0; i<gc->ncuts; i++) {
   
	for(int j=0; j<nCards; j++){
	  ps_reduce<<<gc->pssize1[i], 1024, 0, cs>>> (&gc->cfft[csi][2*j*gc->transform_size], 
						      &(gc->coutps[csi][psofs]), gc->ndxofs[i], gc->fftavg[i]);
	  psofs+=gc->pssize1[i];
        
	  ps_reduce<<<gc->pssize1[i], 1024, 0, cs>>> (&gc->cfft[csi][(2*j+1)*gc->transform_size], 
						      &(gc->coutps[csi][psofs]), gc->ndxofs[i], gc->fftavg[i]);
	  psofs+=gc->pssize1[i];
	}
	//cross spectra
	for(int j = 0; j<nCards*2; j++)
	  for(int k = j+1; k < nCards*2 ; k++){
	    //NEED TO CHECK THAT PARAMETERS ARE ALL CORRECT FOR TWO CARDS AND FOR ONE CARD....
	    ps_X_reduce<<<gc->pssize1[i], 1024, 0, cs>>> (&gc->cfft[csi][j*gc->transform_size], 
							  &gc->cfft[csi][k*gc->transform_size], 
							  &(gc->coutps[csi][psofs]), &(gc->coutps[csi][psofs+gc->pssize1[i]]),
							  gc->ndxofs[i], gc->fftavg[i]);
	    psofs+=2*gc->pssize1[i];
	  }
      }
    }
    else{
      printf("Can only handle 1 or 2 channels\n");
      exit(1);
    }
    cudaEventRecord(gc->eDonePost[csi], cs);
    cudaEventRecord(gc->eDoneCalib[csi], cs);
  }



  if (set->measure_delay || gc->calibrating) {
    cudaEventRecord(gc->eDonePost[csi], cs);
    gc->calib[csi]=true;

    int blocksPerGrid = gc->transform_size / threadsPerBlock;
    C12_Cross <<<blocksPerGrid, threadsPerBlock, 0, cs >>> (&(gc->cfft[csi][0]), 
              &(gc->cfft[csi][gc->transform_size]),
    	      &(gc->cfft[csi][2*gc->transform_size]), 
              &(gc->cfft[csi][3*gc->transform_size]));
    
    int status = cufftSetStream(gc->iplan, cs);
     if(status !=CUFFT_SUCCESS) {
       printf("CUFFTSETSTREAM failed\n");
       exit(1);
     }
        status=cufftExecC2R(gc->iplan, &(gc->cfft[csi][0]), &(gc->cfbuf[csi][0]) );
     if (status!=CUFFT_SUCCESS) {
       printf("CUFFT FAILED\n");
       exit(1);
     } 

    blocksPerGrid = threadsPerBlock;
    int mult=gc->fftsize/blocksPerGrid/threadsPerBlock;
    C12_FindMax_Part1<<<blocksPerGrid,threadsPerBlock,0,cs>>> (&(gc->cfbuf[csi][0]),
	       mult,&(gc->cfbuf[csi][gc->fftsize]),(int*)gc->cbuf[csi][0]);
    C12_FindMax_Part2<<<1,1,0,cs>>>(blocksPerGrid, gc->fftsize, 
		&(gc->cfbuf[csi][gc->fftsize]),(int*)gc->cbuf[csi][0],
                &gc->cmeasured_delay[csi]);


    cudaEventRecord(gc->eDoneCalib[csi], cs);

  } 
  // this is outside so that event gets processed.
  


  cudaEventRecord(gc->eDoneStream[csi], cs);
  
  
  return true;
}
