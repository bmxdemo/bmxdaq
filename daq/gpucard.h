/************e***********************
***********************************
THIS IS A COMPLETE PLACEHOLDER!
***********************************
**********************************/
#pragma once



#include "stdint.h"
#include "settings.h"
#include "writer.h"

#ifdef CUDA_COMPILER

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_REAL cufftReal
#define CUDA_COMPLEX cufftComplex
#define CUDA_STREAM cudaStream_t
#define CUDA_EVENT cudaEvent_t

#else

#define CUDA_REAL void
#define CUDA_COMPLEX void
#define CUDA_STREAM void
#define CUDA_EVENT void

#endif


struct GPUCARD {
  uint8_t **cbuf; // pointer to pointers of GPU sample buffer
  CUDA_REAL **cfbuf; // floats
  CUDA_COMPLEX **cfft; // ffts
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
  CUDA_STREAM *streams; // streams
  int fstream, bstream; // front stream (oldest running), back stream (newest runnig);
  int active_streams; // really needed just at the beginning (when 0)
  CUDA_EVENT *eStart, *eDoneCopy, *eDoneFloatize, *eDoneFFT, *eDonePost, *eDoneCopyBack; //events
  CUDA_REAL ** mean, **cmean; //means of chunks of data to be used for rfi rejection
  int RFIchunkSize; //size of chunk
};


extern "C" {
  void gpuCardInit (GPUCARD *gcard, SETTINGS *set);
  bool gpuProcessBuffer(GPUCARD *gcard, int8_t *buf, WRITER *w, SETTINGS *set);
}
