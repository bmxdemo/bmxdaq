#define CUDA_COMPILE
#include "rfi.h"
#include "gpucard.h"
#undef CUDA_COMPILE
#include "reduction.h"
#include "terminal.h"
#include <cuda.h>
#include <cufft.h>
#include <math.h>
   
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

cufftDoubleReal var(cufftDoubleReal ssquare, cufftDoubleReal mean){
        return ssquare - pow(mean, 2);
}


STATISTIC::STATISTIC(STATISTIC_TYPE type, CUFFT_REAL_RFI ** data, int size, int nStreams): type(type), data(data), size(size) {
	tmean = (cufftDoubleReal **)malloc(nStreams * sizeof(cufftDoubleReal *));
	tsqMean = (cufftDoubleReal **)malloc(nStreams * sizeof(cufftDoubleReal *));
	tvar = (cufftDoubleReal **)malloc(nStreams * sizeof(cufftDoubleReal *));
	trms = (cufftDoubleReal **)malloc(nStreams * sizeof(cufftDoubleReal *));
	nulledCounter = (int **)malloc(nStreams * sizeof(cufftDoubleReal *));
        
	for(int i = 0; i < 2; i++){
	    tmean[i] = (cufftDoubleReal * )malloc(2*sizeof(cufftDoubleReal));
	    tsqMean[i] = (cufftDoubleReal * )malloc(2*sizeof(cufftDoubleReal));
	    tvar[i] = (cufftDoubleReal * )malloc(2*sizeof(cufftDoubleReal));
	    trms[i] = (cufftDoubleReal * )malloc(2*sizeof(cufftDoubleReal));
	    nulledCounter[i] = (int * )malloc(2*sizeof(int));
	}
}
 
//calculate mean, variance, standard dev of the statistic over all chunks
void STATISTIC::getMeanRMS(int csi){
    for(int ch=0; ch<2; ch++){ //for each channel
        tmean[csi][ch] = 0;
        tsqMean[csi][ch] = 0;

	for(int i=ch* size/2; i<(ch+1) * size/2; i++){
	    tmean[csi][ch] += data[csi][i]/(size/2);
	    tsqMean[csi][ch]+=pow(data[csi][i], 2)/(size/2);
	}

	tvar[csi][ch] = var(tsqMean[csi][ch], tmean[csi][ch]);
	trms[csi][ch] = sqrt(tvar[csi][ch]);
    }
}

//number of standard deviations away from mean
float STATISTIC::nSigma(int i, int csi){
    int ch = i/(size/2); 
    return  abs(data[csi][i] - tmean[csi][ch])/trms[csi][ch]; 
}

bool STATISTIC::isOutlier(int i, float nsig, int csi){
     if(nSigma(i, csi) > nsig) return true;
     else return false;
}

void STATISTIC::print(int csi){
    if(type == mean) printf("MEAN: ");
    if(type == variance) printf("VARIANCE: ");
    if(type == absoluteMax) printf("ABSOLUTE MAXIMUM: ");
    tprintfn("CH1 outliers: %d CH2 outliers: %d                                                                  ", nulledCounter[csi][0], nulledCounter[csi][1]);
    tprintfn("CH1 mean/var/rms: %f %f %f CH2 mean/var/rms: %f %f %f", tmean[csi][0], tvar[csi][0], trms[csi][0], tmean[csi][1], tvar[csi][1], trms[csi][1]);
    tprintfn("                                                                                                          ");
}

void rfiInit(RFI * rfi, SETTINGS * s, GPUCARD *gc){
    if(!(OPS_PER_THREAD>0) ||  !((OPS_PER_THREAD & (OPS_PER_THREAD-1)) == 0)){
        printf("Need OPS_PER_THREAD to be a power of 2.\n");
        exit(1);
    }

    rfi->nSigmaNull = s->n_sigma_null;
    rfi->nSigmaWrite = s->n_sigma_write;
    rfi->chunkSize = pow(2, s->log_chunk_size);
    rfi->numChunks = gc->bufsize/rfi->chunkSize; //total number of chunks in all channels
    int opsPerThread = min(rfi->chunkSize, OPS_PER_THREAD);
    int numThreads = min(gc->devProp->maxThreadsPerBlock, rfi->chunkSize/opsPerThread);
    int numBlocks = gc->bufsize/(numThreads * opsPerThread);  //number of blocks needed for first kernel call in parallel reduction algorithms
   
    rfi->statFlags = 0;
    if(s->use_mean_statistic) rfi->statFlags |= RFI_MEAN;
    if(s->use_variance_statistic) rfi->statFlags |= RFI_VARIANCE;
    if(s->use_abs_max_statistic) rfi->statFlags |= RFI_ABS_MAX;
    
    if(rfi->statFlags & RFI_VARIANCE){
        rfi->mean = (cufftReal **) malloc(s->cuda_streams *sizeof(cufftReal*));
        rfi->cmean = (cufftReal **) malloc(s->cuda_streams *sizeof(cufftReal*));
        rfi->sqMean = (cufftReal **)malloc(s->cuda_streams*sizeof(cufftReal*));
        rfi->csqMean = (cufftReal **) malloc(s->cuda_streams *sizeof(cufftReal*));
        rfi->variance = (cufftReal **)malloc(s->cuda_streams*sizeof(cufftReal*));
  
        for(int i=0; i<s->cuda_streams; i++){
            CHK(cudaMallocHost(&rfi->mean[i], rfi->numChunks*sizeof(cufftReal)));
            CHK(cudaMalloc(&rfi->cmean[i], numBlocks*sizeof(cufftReal)));
            CHK(cudaMallocHost(&rfi->sqMean[i], rfi->numChunks*sizeof(cufftReal)));
            CHK(cudaMalloc(&rfi->csqMean[i], numBlocks*sizeof(cufftReal)));
            CHK(cudaMallocHost(&rfi->variance[i], rfi->numChunks*sizeof(cufftReal)));
         }   
    }
    else if(rfi->statFlags & RFI_MEAN){
        rfi->mean = (cufftReal **) malloc(s->cuda_streams *sizeof(cufftReal*));
        rfi->cmean = (cufftReal **) malloc(s->cuda_streams *sizeof(cufftReal*));
  
        for(int i=0; i<s->cuda_streams; i++){
            CHK(cudaMallocHost(&rfi->mean[i], rfi->numChunks*sizeof(cufftReal)));
            CHK(cudaMalloc(&rfi->cmean[i], numBlocks*sizeof(cufftReal)));
         }
    }
    if(rfi->statFlags & RFI_ABS_MAX){
        rfi->absMax = (cufftReal **) malloc(s->cuda_streams *sizeof(cufftReal*));
        rfi->cabsMax = (cufftReal **) malloc(s->cuda_streams *sizeof(cufftReal*));
  
        for(int i=0; i<s->cuda_streams; i++){
            CHK(cudaMallocHost(&rfi->absMax[i], rfi->numChunks*sizeof(cufftReal)));
            CHK(cudaMalloc(&rfi->cabsMax[i], numBlocks*sizeof(cufftReal)));
        }
    }
    rfi->isOutlierNull = (bool **)malloc(s->cuda_streams*sizeof(bool*));
    rfi->isOutlierWrite = (bool **)malloc(s->cuda_streams*sizeof(bool*));
    rfi->numOutliersNulled = (int **)malloc(s->cuda_streams * sizeof(int *));
    rfi->outlierBuf = (int8_t ** )malloc(s->cuda_streams * sizeof(int8_t *));
    rfi->outliersOR = (int *)malloc(s->cuda_streams * sizeof(int));
  
    for(int i=0; i<s->cuda_streams; i++){
        rfi->isOutlierNull[i] = (bool *)malloc(rfi->numChunks * sizeof(bool)); //number of chunks in 1 channel
        rfi->isOutlierWrite[i] = (bool *)malloc(rfi->numChunks * sizeof(bool)); //number of chunks in 1 channel
        rfi->numOutliersNulled[i] = (int *)malloc(gc->nchan * sizeof(int));
	rfi->outlierBuf[i] = (int8_t *)malloc(rfi->chunkSize * sizeof(int8_t));
     }   
  
    rfi->avgOutliersPerChannel = (float *)malloc(gc->nchan*sizeof(float));
    memset(rfi->avgOutliersPerChannel, 0, gc->nchan*sizeof(float));

    if(rfi->statFlags & RFI_MEAN) rfi->statistics.insert(std::pair<STATISTIC_TYPE, STATISTIC> (mean, STATISTIC(mean, rfi->mean, rfi->numChunks, gc->nstreams)));
    if(rfi->statFlags & RFI_VARIANCE) rfi->statistics.insert(std::pair<STATISTIC_TYPE,  STATISTIC>(variance,STATISTIC(variance, rfi->variance, rfi->numChunks, gc->nstreams)));
    if(rfi->statFlags & RFI_ABS_MAX) rfi->statistics.insert(std::pair<STATISTIC_TYPE, STATISTIC> (absoluteMax ,STATISTIC(absoluteMax, rfi->absMax, rfi->numChunks, gc->nstreams)));

}


void collectRFIStatistics(RFI* rfi, GPUCARD * gc, int csi){
    cudaStream_t cs = gc->streams[csi];
    
    if(rfi->statFlags & RFI_MEAN || rfi->statFlags & RFI_VARIANCE)
        getMeans(gc->cfbuf[csi], rfi->mean[csi], rfi->cmean[csi], gc->bufsize, rfi->chunkSize, cs, gc->devProp->maxThreadsPerBlock);
    if(rfi->statFlags & RFI_VARIANCE)
        getMeansOfSquares(gc->cfbuf[csi], rfi->sqMean[csi], rfi->csqMean[csi], gc->bufsize, rfi->chunkSize, cs, gc->devProp->maxThreadsPerBlock);
    if(rfi->statFlags & RFI_ABS_MAX)
        getAbsMax(gc->cfbuf[csi], rfi->absMax[csi], rfi->cabsMax[csi], gc->bufsize, rfi->chunkSize, cs, gc->devProp->maxThreadsPerBlock);
  

    //synchronize so don't use memory before GPU finishes copying it to the CPU
    CHK(cudaStreamSynchronize(cs));
    if(rfi->statFlags & RFI_VARIANCE){
        //calculate variance
	for(int i = 0; i <rfi->numChunks; i++)
	    rfi->variance[csi][i] = var(rfi->sqMean[csi][i], rfi->mean[csi][i]);
    }
    for(std::pair<STATISTIC_TYPE, STATISTIC> s: rfi->statistics){
	s.second.getMeanRMS(csi);
	s.second.nulledCounter[csi][0] = 0; s.second.nulledCounter[csi][1] = 0;
    }
}

int hammingWeight(bool * a, bool * b, int size){
	int n = 0;
	for(int i = 0; i< size; i++)
	    if(a[i] == 1 || b[i] == 1) n++;
	return n;
}


void nullRFI(RFI* rfi, GPUCARD * gc, int csi, WRITER * wr){
    if(rfi->nSigmaNull == 0) return;
    
    memset(rfi->isOutlierNull[csi], 0, rfi->numChunks*sizeof(bool)); //reset outlier flags to 0
    int chunksPerChannel = rfi->numChunks/2;
    rfi->numOutliersNulled[csi][0] = 0;   rfi->numOutliersNulled[csi][1] = 0;
    
    for(int i = 0; i < rfi->numChunks; i++)
        for(std::pair<STATISTIC_TYPE, STATISTIC> s: rfi->statistics)
	    if(s.second.isOutlier(i, rfi->nSigmaNull, csi)){
		s.second.nulledCounter[csi][i/chunksPerChannel]++;
		if(rfi->isOutlierNull[csi][i] == false){
		    rfi->isOutlierNull[csi][i] = true;
		    rfi->numOutliersNulled[csi][i/chunksPerChannel]++;
		    CHK(cudaMemsetAsync(&(gc->cfbuf[csi][i*rfi->chunkSize]), 0, rfi->chunkSize, gc->streams[csi])); //null out outlier
		}
	     }

    rfi->outliersOR[csi] = hammingWeight(rfi->isOutlierNull[csi], rfi->isOutlierNull[csi] + chunksPerChannel, chunksPerChannel);
    //calculate approximate average of outliers per channel per sample (approximate because using wr->counter which might be a bit behind)
    for(int i=0; i <gc->nchan; i++)
	rfi->avgOutliersPerChannel[i]= (rfi->avgOutliersPerChannel[i]*wr->counter + rfi->numOutliersNulled[csi][i])/(wr->counter+1);
    
    tprintfn("                                                                                                               ");
    tprintfn("RFI analysis:                                                                                                  ");
    for(std::pair<STATISTIC_TYPE, STATISTIC> s: rfi->statistics)
	s.second.print(csi);
    tprintfn("TOTAL: CH1 outliers: %d CH2 outliers: %d", rfi->numOutliersNulled[csi][0], rfi->numOutliersNulled[csi][1]);
    tprintfn("CH1 average outliers: %f CH2 average outliers: %f", rfi->avgOutliersPerChannel[0], rfi->avgOutliersPerChannel[1]); 
}

void writeRFI(RFI* rfi, GPUCARD * gc, int csi, WRITER * wr, int8_t * buf){
    if(rfi->nSigmaWrite == 0) return;
    memset(rfi->isOutlierWrite[csi], 0, rfi->numChunks*sizeof(bool)); //reset outlier flags to 0
    float sigs [STAT_COUNT_MINUS_ONE + 1] = {0};    
    for(int i = 0; i < rfi->numChunks; i++)
        for(std::pair<STATISTIC_TYPE, STATISTIC> s: rfi->statistics)
	    if(s.second.isOutlier(i, rfi->nSigmaWrite, csi)){
	       sigs[s.second.type] = s.second.nSigma(i, csi); //print out how many sigma away from mean
	       if(rfi->isOutlierWrite[csi][i] == false){
		   rfi->isOutlierWrite[csi][i] = true;
		   int ch = i/(rfi->numChunks/2);
   	           for(uint32_t j =0; j<rfi->chunkSize; j++)
	              rfi->outlierBuf[csi][j] = buf[2*(i%2 * rfi->chunkSize + j) + ch]; //deinterleave data in order to write out to file 
	           writerWriteRFI(wr, rfi->outlierBuf[csi], i%2 , ch, sigs );
	       }
	    }
}

