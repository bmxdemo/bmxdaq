#pragma once

#include "writer.h"

#ifdef CUDA_COMPILE
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#define CUFFT_REAL_RFI cufftReal

#else
#define CUFFT_REAL_RFI void
#endif

struct GPUCARD; //forward declaration

struct RFI{
  float nSigmaNull; //number of standard deviations used to determine outliers to null out, -1 for none
  float nSigmaWrite; //number of standard deviations used to determine outliers to write to file, -1 for none
  CUFFT_REAL_RFI ** mean, **cmean, **sqMean, **csqMean, **variance, **absMax, **cabsMax; //statistics for rfi rejection (mean, mean sum of squares, variance)  
  int chunkSize; //size of chunk
  int8_t * outlierBuf; //holds outlier data to print to file
  int ** isOutlier; //array of flags detemining if chunk is outlier or not
  float * avgOutliersPerChannel; //average number of outlier chunks per channel per sample since program began running
  int ** numOutliersNulled; //number of outliers nulled per channel
  int * outliersOR; //number of outliers obtained by a logical OR on the arrays of outlier flags from the different channels

};

void rfiInit(RFI * rfi, SETTINGS * s, GPUCARD *gc);
void detectRFI(RFI * rfi, GPUCARD *gc, int csi, WRITER * wr);
//void nullRFI();
//void writeRFI();
//void cleanupRFI();




