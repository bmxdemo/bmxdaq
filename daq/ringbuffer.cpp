#include "digicard.h"
#include "digicardalloc.h"
#include "ringbuffer.h"
#include "terminal.h"
#include "stdio.h"
#include "time.h"
#include "unistd.h"

void ringbufferInit(RINGBUFFER *rb, SETTINGS *set, DIGICARD *dc){
  if (set->ringbuffer_size<=0) return;
  printf ("==========================\n");
  rb->bufsize=dc->lNotifySize;
  rb->ncards=dc->num_cards;
  rb->dumping=0;
  rb->filling[0]=rb->filling[1]=false;
  rb->fillremain=set->ringbuffer_size;
  rb->force=set->ringbuffer_force;
  strcpy(rb->fname_pattern,set->ringbuffer_output_pattern);

  time_t rawtime;
  time ( &rawtime );
  struct tm *ti = gmtime ( &rawtime );
  snprintf(rb->filename, MAXFNLEN,rb->fname_pattern, ti->tm_year - 100 , ti->tm_mon + 1, 
	  ti->tm_mday, ti->tm_hour, ti->tm_min, ti->tm_sec);


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
      //int16* p;
      //digiCardAlloc(p, rb->bufsize);
      if (p==NULL){
	printf ("Allocation failure in ringbuffer.\n");
	exit(1);
      }
      rb->buffer[icard*MAXCHUNKS+i]=(int8_t*)p;
      //printf ("%i %i %li\n",icard,i,rb->buffer[icard*MAXCHUNKS+i]);
    }
  }
  printf ("Ringbuffer allocated for %i cards x %i chunks x %i bytes\n",
	  rb->ncards, rb->num_chunks, rb->bufsize);
}



void copyAction(RINGBUFFER *rb, int cardnum){
// do copy
//...
  int8_t* dest=rb->buffer[cardnum*MAXCHUNKS+rb->cur_chunk[cardnum]];
  memcpy(dest,rb->src[cardnum],rb->bufsize);
  rb->cur_chunk[cardnum]= (++rb->cur_chunk[cardnum])%(rb->num_chunks);
  rb->filling[cardnum]=false;
  if ((cardnum==0) && (rb->fillremain>0)) rb->fillremain--;
}

void copyAction0(RINGBUFFER &rb) {copyAction(&rb,0);}
void copyAction1(RINGBUFFER &rb){copyAction(&rb,1);}



void fillRingBuffer(RINGBUFFER *rb, int8_t* src[2]) {
  if (rb->num_chunks==0) return;
  if (rb->dumping) return;
  if (rb->dumpthread.joinable()) rb->dumpthread.join();

  // this makes the ringbuffer wait. Seem to keep the fill at 100% without
  // issues but perhaps revisit if neccessary.
  if (rb->force) {
    for (int cardnum=0; cardnum<rb->ncards; cardnum++) 
      if (rb->thread[cardnum].joinable()) rb->thread[cardnum].join();
  }

  if (rb->filling[0] || rb->filling[1]) {
    rb->fillremain=rb->num_chunks;
  } else {
    for (int cardnum=0; cardnum<rb->ncards; cardnum++) {
      rb->filling[cardnum]=true;
      rb->src[cardnum]=src[cardnum];
      if (rb->thread[cardnum].joinable()) rb->thread[cardnum].join();
      if (cardnum==0)
	rb->thread[cardnum] = std::thread(copyAction0,std::ref(*rb));
      else
	rb->thread[cardnum] = std::thread(copyAction1,std::ref(*rb));
    }
  }
}


void dumpAction(RINGBUFFER &rbr) {
  RINGBUFFER *rb=&rbr;
  RINGBUFFERHEADER head;
  head.ncards=rb->ncards;
  head.totbufsize=rb->bufsize*rb->num_chunks;
  int totsize=rb->num_chunks*rb->ncards;
  FILE *f=fopen(rb->filename,"wb");
  fwrite(&head,sizeof(RINGBUFFERHEADER),1,f);
  for (int icard=0; icard<rb->ncards; icard++) {
    //printf ("allocating buffers: %i %i\n",icard,rb->num_chunks);
    for (int i=0;i<rb->num_chunks;i++) {
      int ofs=(rb->cur_chunk[icard]+i)%(rb->num_chunks);
      fwrite (rb->buffer[icard*MAXCHUNKS+ofs],rb->bufsize,1,f);
      rb->dumpercent=int((icard*rb->num_chunks+i)*100/totsize);
      usleep(1000);
    }
  }
  fclose(f);
  rb->dumping=false;
}

void dumpRingBuffer(RINGBUFFER *rb) {
  if (rb->fillremain>0) return;
  time_t rawtime;
  time ( &rawtime );
  struct tm *ti = gmtime ( &rawtime );
  snprintf(rb->filename, MAXFNLEN,rb->fname_pattern, ti->tm_year - 100 , ti->tm_mon + 1, 
	  ti->tm_mday, ti->tm_hour, ti->tm_min, ti->tm_sec);
  rb->dumping=true;
  rb->dumpercent=0;
  rb->dumpthread=std::thread(dumpAction, std::ref(*rb));
}



void printInfoRingBuffer(RINGBUFFER *rb, TWRITER *tw) {
  tprintfn (tw,0,"Ringbuffer: ");
  if (rb->num_chunks==0) tprintfn(tw,1,"disabled.");
  if (rb->dumping) tprintfn (tw,1, "Dumping %s ... %i%% ",rb->filename, rb->dumpercent);
  else {
    if (rb->ncards==2) 
      tprintfn (tw,0," Card 1: %03d   Card 2:  %03d ",rb->cur_chunk[0], rb->cur_chunk[1]);
    else
      tprintfn (tw,0," Card 1: %02d ",rb->cur_chunk[0]);
    tprintfn (tw,1,"Fill: %03d %%",int((rb->num_chunks-rb->fillremain)*100/rb->num_chunks));
  }
}

void ringbufferCleanUp(RINGBUFFER *rb) {
  if (rb->thread[0].joinable()) rb->thread[0].join();
  if (rb->thread[1].joinable()) rb->thread[1].join();
}
