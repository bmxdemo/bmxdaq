#pragma once

#include "settings.h"
#include <vector>
#include <map>

#ifdef CUDA_COMPILE
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#define CUFFT_REAL_RFI cufftReal
#define CUFFT_DOUBLE_REAL_RFI cufftDoubleReal

#else
#define CUFFT_REAL_RFI void
#define CUFFT_DOUBLE_REAL_RFI void 
#endif

#define RFI_MEAN 1
#define RFI_VARIANCE 2
#define RFI_ABS_MAX 4

//forward declarations to prevent two header files from including the other
struct GPUCARD; 
struct WRITER; 

enum STATISTIC_TYPE{mean, variance, absoluteMax, STAT_COUNT_MINUS_ONE = absoluteMax};

class STATISTIC {
    public:
    STATISTIC_TYPE type;
    CUFFT_REAL_RFI ** data;
    int size; //number of chunks in data
    CUFFT_DOUBLE_REAL_RFI ** tmean;
    CUFFT_DOUBLE_REAL_RFI ** tsqMean;
    CUFFT_DOUBLE_REAL_RFI ** tvar;
    CUFFT_DOUBLE_REAL_RFI ** trms;
    int ** nulledCounter;

    STATISTIC(STATISTIC_TYPE type, CUFFT_REAL_RFI ** data, int size, int nStreams);
    void getMeanRMS(int csi);
    bool isOutlier(int i, float nsig, int csi);
    float  nSigma(int i, int csi);
    void print(int csi);
};

struct RFI{
  float nSigmaNull; //number of standard deviations used to determine outliers to null out, -1 for none
  float nSigmaWrite; //number of standard deviations used to determine outliers to write to file, -1 for none
  int chunkSize; //size of chunk
  int numChunks; //number of chunks in buffer
  int8_t ** outlierBuf; //holds outlier data to print to file
  bool ** isOutlierNull; //array of flags detemining if chunk is outlier or not
  bool ** isOutlierWrite; //array of flags detemining if chunk is outlier or not
  int * outliersOR; //number of outliers obtained by a logical OR on the arrays of outlier flags from the different channels
  float * avgOutliersPerChannel; //average number of outlier chunks per channel per sample since program began running
  int ** numOutliersNulled; //number of outliers nulled per channel
  
  std::map<STATISTIC_TYPE, STATISTIC> statistics;
  unsigned int statFlags; //bit flags indicating which statistics to use

  CUFFT_REAL_RFI ** mean, **cmean, **sqMean, **csqMean, **variance, **absMax, **cabsMax; //statistics for rfi rejection (mean, mean sum of squares, variance)  
};


void rfiInit(RFI * rfi, SETTINGS * s, GPUCARD *gc);
void collectRFIStatistics(RFI* rfi, GPUCARD * gc, int csi);
void nullRFI(RFI* rfi, GPUCARD * gc, int csi, WRITER * wr);
void writeRFI(RFI* rfi, GPUCARD * gc, int csi, WRITER * wr, int8_t * buf);
//void cleanupRFI();


    

