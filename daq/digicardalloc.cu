#include <cuda.h>
#include "digicardalloc.h"

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

void digiCardAlloc(int16* & data, size_t size){
    //ALLOC(data, size);
    if(cudaMallocHost(&data, size)!= cudaSuccess){
	printf("Digitizer memory allocation failed\n");
	printf ("Requested %li bytes = %i Gb.\n",size, size/1024/1024/1024);
	exit(1);
    }
}
     
void digiCardFree(int16* & data){
    if(cudaFreeHost(data)!= cudaSuccess){
	printf("Freeing of digitizer memory failed\n");
	exit(1);
    }
}
