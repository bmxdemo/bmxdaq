/************e***********************
***********************************
THIS IS A COMPLETE PLACEHOLDER!
***********************************
**********************************/
#pragma once



#include "stdint.h"
#include "settings.h"
#include "writer.h"

struct GPUCARD {
  void **cbuf; // pointer to pointers of GPU sample buffer
  void **cfbuf; // floats
  void **cfft; // ffts
  void **coutps; // output power spectra
  float *outps;
  int nchan; // nchannels
  uint32_t fftsize; // fft size
  uint32_t bufsize; // buffer size in bytes
  int fftavg;
  int threads; // threads to use
  int plan;
  int nstreams;
  int pssize1; // size of one power spectrum (in indices)
  int pssize; // size of how many we produce
  int ndxofs; // which offset we start averaging
  void *streams; // streams
  int fstream, bstream; // front stream (oldest running), back stream (newest runnig);
  int active_streams; // really needed just at the beginning (when 0)
  void *eStart, *eDoneCopy, *eDoneFloatize, *eDoneFFT, *eDonePost, *eDoneCopyBack; //events

  
};


extern "C" {
  void gpuCardInit (GPUCARD *gcard, SETTINGS *set);
  bool gpuProcessBuffer(GPUCARD *gcard, int8_t *buf, WRITER *w);
}
