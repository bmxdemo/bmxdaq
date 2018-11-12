#include "digicard.h"
#include "ringbuffer.h"
#include "terminal.h"

void ringbufferInit(RINGBUFFER *rb, SETTINGS *set, DIGICARD *dc){
  printf ("==========================\n");
  rb->bufsize=dc->lNotifySize;
  rb->ncards=dc->num_cards;
  rb->dumping=0;
  strcpy(rb->fname_pattern,set->ringbuffer_output_pattern);
  if (set->ringbuffer_size>MAXCHUNKS) {
    printf ("Ring buffer cannot be that big.\n");
    exit(1);
  }
  rb->num_chunks=set->ringbuffer_size;
  rb->cur_chunk[0]=rb->cur_chunk[1]=0;
  //printf ("allocating buffers:\n");
  for (int icard=0; icard<rb->ncards; icard++) {
    //printf ("allocating buffers: %i %i\n",icard,rb->num_chunks);
    for (int i=0;i<rb->num_chunks;i++) {
      //printf ("%i %i %li\n",icard,i,rb->buffer[icard*MAXCHUNKS+i]);
      int8_t *p=(int8_t*)malloc(rb->bufsize);
      if (p==NULL){
	printf ("Allocation failure in ringbuffer.\n");
	exit(1);
      }
      rb->buffer[icard*MAXCHUNKS+i]=p;
      //printf ("%i %i %li\n",icard,i,rb->buffer[icard*MAXCHUNKS+i]);
    }
  }
  printf ("Ringbuffer allocated for %i cards x %i chunks x %i bytes\n",
	  rb->ncards, rb->num_chunks, rb->bufsize);
}


void dumpRingBuffer(RINGBUFFER *rb, int8_t inbuf) {
  
}

void copyAction(RINGBUFFER *rb, int cardnum, int8_t* src){
// do copy
//...
  rb->filling[cardnum]=false;
  int8_t* dest=rb->buffer[cardnum*MAXCHUNKS+rb->cur_chunk[cardnum]];
  //printf("\n COPYING: %li %li %i %i\n", dest, src,cardnum, rb->cur_chunk[cardnum]);
  //  memcpy(dest,src,rb->bufsize);
  rb->cur_chunk[cardnum]= (++rb->cur_chunk[cardnum])%(rb->num_chunks);
}

void copyAction0(RINGBUFFER &rb, int8_t& src){copyAction(&rb,0,&src);}
void copyAction1(RINGBUFFER &rb, int8_t& src){copyAction(&rb,1,&src);}



void fillRingBuffer(RINGBUFFER *rb, int cardnum, int8_t* src) {
  //if (rb->filling[cardnum]) {
  //printf("Ring buffer can't keep up!\n");
  //exit(1);
  // }
  rb->filling[cardnum]=true;
  if (rb->thread[cardnum].joinable()) rb->thread[cardnum].join();
  if (cardnum==0)
    rb->thread[cardnum] = std::thread(copyAction0,std::ref(*rb), std::ref(*src));
  else
    rb->thread[cardnum] = std::thread(copyAction1,std::ref(*rb), std::ref(*src));
}

void printInfoRingBuffer(RINGBUFFER *rb, TWRITER *tw) {
tprintfn (tw,0,"Ringbuffer: ");
 if (rb->dumping) tprintfn (tw,1, "Dumping to disk: ...");
 else if (rb->ncards==2) 
   tprintfn (tw,1," Card 1: %i   Card 2:  %i ",rb->cur_chunk[0], rb->cur_chunk[1]);
 else
   tprintfn (tw,1," Card 1: %i ",rb->cur_chunk[0]);
}

void ringbufferCleanUp(RINGBUFFER *rb) {
  if (rb->thread[0].joinable()) rb->thread[0].join();
  if (rb->thread[1].joinable()) rb->thread[1].join();
}
