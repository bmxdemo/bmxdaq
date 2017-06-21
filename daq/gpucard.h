/************e***********************
***********************************
THIS IS A COMPLETE PLACEHOLDER!
***********************************
**********************************/
#pragma once



#include "stdint.h"
#include "settings.h"
#include "writer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

struct GPUCARD {
  int8_t **cbuf; // pointer to pointers of GPU sample buffer
  cufftReal **cfbuf; // floats
  cufftComplex **cfft; // ffts
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
  cudaStream_t *streams; // streams
  int fstream, bstream; // front stream (oldest running), back stream (newest runnig);
  int active_streams; // really needed just at the beginning (when 0)
  cudaEvent_t *eStart, *eDoneCopy, *eDoneFloatize, *eDoneFFT, *eDonePost, *eDoneCopyBack; //events
  cufftReal ** mean, **cmean, **sqMean, **csqMean, **variance; //statistics for rfi rejection (mean, mean sum of squares, variance) 
  int RFIchunkSize; //size of chunk
};


extern "C" {
  void gpuCardInit (GPUCARD *gcard, SETTINGS *set);
  bool gpuProcessBuffer(GPUCARD *gcard, int8_t *buf, WRITER *w, SETTINGS *set);
}
