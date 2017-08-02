#define CUDA_COMPILE
#include "rfi.h"
#include "gpucard.h"
#undef CUDA_COMPILE
#include "reduction.h"
#include "terminal.h"
#include <cuda.h>
#include <cufft.h>
   
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



cufftDoubleReal variance(cufftDoubleReal ssquare, cufftDoubleReal mean){
        return ssquare - pow(mean, 2);
}

void rfiInit(RFI * rfi, SETTINGS * s, GPUCARD *gc){
  rfi->nSigmaNull = s->n_sigma_null;
  rfi->nSigmaWrite = s->n_sigma_write;
  rfi->chunkSize = pow(2, s->log_chunk_size);
  int numThreads = min(gc->devProp->maxThreadsPerBlock, rfi->chunkSize/OPS_PER_THREAD);
  int numBlocks = gc->bufsize/(numThreads * OPS_PER_THREAD);  //number of blocks needed for first kernel call in parallel reduction algorithms
 
   
  rfi->mean = (cufftReal **) malloc(s->cuda_streams *sizeof(cufftReal*));
  rfi->cmean = (cufftReal **) malloc(s->cuda_streams *sizeof(cufftReal*));
  rfi->sqMean = (cufftReal **)malloc(s->cuda_streams*sizeof(cufftReal*));
  rfi->csqMean = (cufftReal **) malloc(s->cuda_streams *sizeof(cufftReal*));
  rfi->variance = (cufftReal **)malloc(s->cuda_streams*sizeof(cufftReal*));
  rfi->absMax = (cufftReal **) malloc(s->cuda_streams*sizeof(cufftReal*));
  rfi->cabsMax = (cufftReal **) malloc(s->cuda_streams*sizeof(cufftReal*));
  rfi->isOutlier = (int **)malloc(s->cuda_streams*sizeof(int*));
  rfi->outlierBuf = (int8_t * )malloc(rfi->chunkSize);
  rfi->numOutliersNulled = (int **)malloc(s->cuda_streams * sizeof(int *));
  rfi->outliersOR = (int *)malloc(s->cuda_streams * sizeof(int));

  for(int i=0; i<s->cuda_streams; i++){
      CHK(cudaMalloc(&rfi->cmean[i], numBlocks*sizeof(cufftReal)));
      CHK(cudaMallocHost(&rfi->mean[i], gc->bufsize/rfi->chunkSize*sizeof(cufftReal)));
      CHK(cudaMalloc(&rfi->csqMean[i], numBlocks*sizeof(cufftReal)));
      CHK(cudaMallocHost(&rfi->sqMean[i], gc->bufsize/rfi->chunkSize*sizeof(cufftReal)));
      CHK(cudaMallocHost(&rfi->variance[i], gc->bufsize/rfi->chunkSize*sizeof(cufftReal))); //total number of chunks in all channels
      CHK(cudaMalloc(&rfi->cabsMax[i], numBlocks*sizeof(cufftReal)));
      CHK(cudaMallocHost(&rfi->absMax[i], gc->bufsize/rfi->chunkSize*sizeof(cufftReal)));
      rfi->isOutlier[i] = (int *)malloc(gc->bufsize/rfi->chunkSize/gc->nchan * sizeof(int)); //number of chunks in 1 channel
      rfi->numOutliersNulled[i] = (int *)malloc(gc->nchan * sizeof(int));
   }   

  rfi->avgOutliersPerChannel = (float *)malloc(gc->nchan*sizeof(float));
  memset(rfi->avgOutliersPerChannel, 0, gc->nchan*sizeof(float));
}



void detectRFI(RFI* rfi, GPUCARD * gc, int csi, WRITER * wr){
    cudaStream_t cs = gc->streams[csi];
    int numChunks = gc->bufsize/rfi->chunkSize; //total number of chunks in all channels
    rfi->outliersOR[csi] = 0; //number of outliers obtained by a logical OR on the arrays of outlier flags from the different channels
    
    getMeans(gc->cfbuf[csi], rfi->mean[csi], rfi->cmean[csi], gc->bufsize, rfi->chunkSize, cs, gc->devProp->maxThreadsPerBlock);
    getMeansOfSquares(gc->cfbuf[csi], rfi->sqMean[csi], rfi->csqMean[csi], gc->bufsize, rfi->chunkSize, cs, gc->devProp->maxThreadsPerBlock);
    getAbsMax(gc->cfbuf[csi], rfi->absMax[csi], rfi->cabsMax[csi], gc->bufsize, rfi->chunkSize, cs, gc->devProp->maxThreadsPerBlock);

    cufftDoubleReal tmean[2]={0}, tsqMean[2]={0}, tvar[2], trms[2]; //for 2 channels. Note: double precision is neccesary or results will be incorrect!
    memset(rfi->isOutlier[csi], 0, numChunks/gc->nchan*sizeof(int)); //reset outlier flags to 0

    cufftReal ** statistic = rfi->variance; //desired statistic(s) to use to determine outliers

    //synchronize so don't use memory before GPU finishes copying it to the CPU
    CHK(cudaStreamSynchronize(cs));

    for(int ch=0; ch<2; ch++){ //for each channel
	rfi->numOutliersNulled[csi][ch] = 0;

	//calculate mean, variance, standard dev of the statistic over all chunks
	for(int i=ch* numChunks/2; i<(ch+1) * numChunks/2; i++){
	    rfi->variance[csi][i] = variance(rfi->sqMean[csi][i], rfi->mean[csi][i]);
	    tmean[ch] += statistic[csi][i]/(numChunks/2);
	    tsqMean[ch]+=pow(statistic[csi][i], 2)/(numChunks/2);
	}

	tvar[ch] = variance(tsqMean[ch], tmean[ch]);
	trms[ch] = sqrt(tvar[ch]);

	//handle rfi
	for(int i=ch* numChunks/2; i<(ch+1) * numChunks/2; i++){
	    float nSigma = abs(statistic[csi][i] - tmean[ch])/trms[ch]; //number of standard deviations away from mean
	    if(nSigma > rfi->nSigmaNull){
       	       rfi->numOutliersNulled[csi][ch]++;
	       //mimic logical OR of flagged chunks in each channel
	       if(rfi->isOutlier[csi][i%2] == 0){//other channel didn't flag this chunk
		    rfi->isOutlier[csi][i%2] = 1; //flag as outlier
		    rfi->outliersOR++;
	        }
	        if(rfi->nSigmaNull > -1) CHK(cudaMemsetAsync(&(gc->cfbuf[csi][i*rfi->chunkSize]), 0, rfi->chunkSize, cs)); //zero out outliers for FFT

	        //for(uint32_t j =0; j<rfi->chunkSize; j++)
		  //  rfi->outlierBuf[j] = buf[2*(i%2 * rfi->chunkSize + j) + ch]; //deinterleave data in order to write out to file 

	        //Write outlier to file
	        if(rfi->nSigmaWrite > -1 && nSigma > rfi->nSigmaWrite)
		    writerWriteRFI(wr, rfi->outlierBuf, i%2 , ch, nSigma);
	   }
       }
    }
    //calculate approximate average of outliers per channel per sample (approximate because using wr->counter which might be a bit behind)
    int n = wr->counter;
    for(int i=0; i <gc->nchan; i++)
	rfi->avgOutliersPerChannel[i]= (rfi->avgOutliersPerChannel[i]*n + rfi->numOutliersNulled[csi][i])/(n+1);

    tprintfn(" ");
    tprintfn("RFI analysis: ");
    tprintfn("CH1 mean/var/rms: %f %f %f CH2 mean/var/rms: %f %f %f", tmean[0], tvar[0], trms[0], tmean[1], tvar[1], trms[1]);
    tprintfn("CH1 outliers: %d CH2 outliers: %d", rfi->numOutliersNulled[csi][0], rfi->numOutliersNulled[csi][1]);
    tprintfn("CH1 average outliers: %f CH2 average outliers: %f", rfi->avgOutliersPerChannel[0], rfi->avgOutliersPerChannel[1]); 
    cudaEventRecord(gc->eDoneRFI[csi],cs);

}






