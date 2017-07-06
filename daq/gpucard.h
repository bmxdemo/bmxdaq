/************e***********************
***********************************
THIS IS A COMPLETE PLACEHOLDER!
***********************************
**********************************/
#pragma once



#include "stdint.h"
#include "settings.h"
#include "writer.h"

#ifdef CUDA_COMPILE

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define CUFFT_REAL cufftReal
#define CUFFT_COMPLEX cufftComplex
#define CUDA_STREAM_T cudaStream_t
#define CUDA_EVENT_T cudaEvent_t
#define CUDA_DEVICE_PROP cudaDeviceProp

#else

#define CUFFT_REAL void
#define CUFFT_COMPLEX void
#define CUDA_STREAM_T void
#define CUDA_EVENT_T void
#define CUDA_DEVICE_PROP void

#endif


struct GPUCARD {
  CUDA_DEVICE_PROP * devProp; //gpu device properties  
  int8_t **cbuf; // pointer to pointers of GPU sample buffer
  CUFFT_REAL **cfbuf; // floats
  CUFFT_COMPLEX **cfft; // ffts
  float **coutps; // output power spectra
  float *outps;
  int nchan; // nchannels
  uint32_t fftsize; // fft size
  uint32_t bufsize; // buffer size in bytes
  int ncuts; // number of ps cuts
  int fftavg[MAXCUTS];
  int pssize1[MAXCUTS]; // size of one power spectrum (in indices)
  int pssize[MAXCUTS]; // size of how many we produce (in indices)
  int tot_pssize; // total size (in indices)
  int ndxofs[MAXCUTS]; // which offset we start averaging
  int threads; // threads to use
  int plan;
  int nstreams;
  CUDA_STREAM_T *streams; // streams
  int fstream, bstream; // front stream (oldest running), back stream (newest runnig);
  int active_streams; // really needed just at the beginning (when 0)
  CUDA_EVENT_T *eStart, *eDoneCopy, *eDoneFloatize, *eDoneRFI,  *eDoneFFT, *eDonePost, *eBeginCopyBack, *eDoneCopyBack; //events
  CUFFT_REAL ** mean, **cmean, **sqMean, **csqMean, **variance; //statistics for rfi rejection (mean, mean sum of squares, variance) 
  int chunkSize; //size of chunk
  int nsigma; //number of standard deviations used to mark outliers
  int8_t * outlierBuf; //holds outlier data to print to file
  int ** isOutlier; //array of flags detemining if chunk is outlier or not
};


extern "C" {
  void gpuCardInit (GPUCARD *gcard, SETTINGS *set);
  bool gpuProcessBuffer(GPUCARD *gcard, int8_t *buf, WRITER *w, SETTINGS *set);
}
