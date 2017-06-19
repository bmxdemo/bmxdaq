#include <cuda.h>
#include "digicard.h"

//#ifdef USE_DIGICARD_ALLOC 
//#define ALLOC(data, size) \
//    (data) = (int16*) pvAllocMemPageAligned ((uint64) (size);	
//#else
//#define ALLOC(data, size) \
//     if(cudaMallocHost(&data, size)!=cudaSuccess){\
//	 printf("Error allocating memory for digitizer buffer.\n");\
//         exit(1);\
//     }
//#endif

void digiCardAlloc(int16* & data, int32 size){
    //ALLOC(data, size);
    if(cudaMallocHost(&data, size)!= cudaSuccess){
	printf("Digitizer memory allocation failed\n");
	exit(1);
    }
}
     
