/***********************************
***********************************
THIS IS A COMPLETE PLACEHOLDER!
***********************************
**********************************/
#pragma once



#include "stdint.h"
#include "settings.h"
#include "writer.h"
#include "terminal.h"

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
  int8_t ***cbuf; // original buffer data as bytes before it is converted to floats
  CUFFT_REAL **cfbuf; // floats
  CUFFT_COMPLEX **cfft; // ffts
  float **coutps; // output power spectra
  float *outps;
  int nchan; // nchannels
  uint32_t fftsize; // fft size for one channel
  uint32_t bufsize; // buffer size in bytes
  uint32_t transform_size; // transform size for one channel
  int ncuts; // number of ps cuts
  int fftavg[MAXCUTS];
  int pssize1[MAXCUTS]; // size of one power spectrum (in indices)
  int pssize[MAXCUTS]; // size of how many we produce (in indices)
  int tot_pssize; // total size (in indices)
  int ndxofs[MAXCUTS]; // which offset we start averaging
  int threads; // threads to use
  int plan, iplan; // forward and inverse plans
  int nstreams;
  CUDA_STREAM_T *streams; // streams
  int *cmeasured_delay;
  int last_measured_delay;
  int fstream, bstream; // front stream (oldest running), back stream (newest runnig);
  int active_streams; // really needed just at the beginning (when 0)
  CUDA_EVENT_T *eStart, *eDoneCopy, *eDoneFloatize,  *eDoneFFT, *eDonePost, *eDoneCalib;
  CUDA_EVENT_T *eDoneStream; //events
};


extern "C" {
  void gpuCardInit (GPUCARD *gcard, SETTINGS *set);
  int  gpuProcessBuffer(GPUCARD *gcard, int8_t ** buf, int8_t ** prevbuf, 
			WRITER *w, TWRITER *t, SETTINGS *set);
}
