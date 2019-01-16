#include "writer.h"
#include "string.h"
#include "time.h"
#include <unistd.h>
#include "stdlib.h"
#include <math.h>
#include <assert.h>

void closeAndRename(WRITER *writer) {
	fclose(writer->fPS);
	rename(writer->tafnamePS,writer->afnamePS);
  if(writer->rfiOn){
	  fclose(writer->fRFI);
	  rename(writer->tafnameRFI,writer->afnameRFI);
  }
}

void maybeReOpenFile(WRITER *writer, SETTINGS *set, bool first=false) {
  time_t rawtime;   
  time ( &rawtime );
  struct tm *ti = gmtime ( &rawtime );
  
  if (first || ((ti->tm_min%writer->new_file_every==0) && writer->reopen)) {
    if (!first) closeAndRename(writer);

    writer->counter =0; //reset sample counter to 0
    sprintf(writer->afnamePS,writer->fnamePS, ti->tm_year - 100 , ti->tm_mon + 1, 
	    ti->tm_mday, ti->tm_hour, ti->tm_min);
    sprintf(writer->tafnamePS,"%s.new",writer->afnamePS);
    writer->fPS=fopen(writer->tafnamePS,"wb");
    
    if (writer->fPS==NULL) {
      printf ("CANNOT OPEN FILE:%s",writer->tafnamePS);
      exit(1);
    }
    
    writer->headerPS.bufdelay[0]=set->bufdelay[0];
    writer->headerPS.bufdelay[1]=set->bufdelay[1];
    writer->headerPS.delay[0]=set->delay[0];
    writer->headerPS.delay[1]=set->delay[1];

    fwrite(&writer->headerPS, sizeof(BMXHEADER),1, writer->fPS);

    if(writer->rfiOn){
      sprintf(writer->afnameRFI,writer->fnameRFI, ti->tm_year - 100 , ti->tm_mon + 1, 
	      ti->tm_mday, ti->tm_hour, ti->tm_min);
      sprintf(writer->tafnameRFI,"%s.new",writer->afnameRFI);
      writer->fRFI=fopen(writer->tafnameRFI,"wb");
      if (writer->fRFI==NULL) {
        printf ("CANNOT OPEN FILE:%s",writer->tafnameRFI);
        exit(1);
      }
      fwrite(&writer->headerRFI, sizeof(RFIHEADER), 1, writer->fRFI);    
    }
    writer->reopen=false;
  }

  if (ti->tm_min%writer->new_file_every==1)  writer->reopen=true;
  
}

void writerInit(WRITER *writer, SETTINGS *s) {
  printf ("==========================\n");
  strcpy(writer->fnamePS,s->ps_output_pattern);
  strcpy(writer->fnameRFI,s->rfi_output_pattern);
  writer->enabled=false;
  writer->new_file_every=s->new_file_every;
  writer->headerPS.cardMask=s->card_mask;
  writer->headerPS.nChannels=1+(s->channel_mask==3);
  writer->headerPS.sample_rate=s->sample_rate;
  writer->headerPS.fft_size=s->fft_size;
  writer->rfiOn=true;
  //writer->headerPS.ADC_range = s->ADC_range;
  //initialize statistics array
  //writer->headerPS.statistics[mean] = s->use_mean_statistic? 1: 0;
  //writer->headerPS.statistics[variance] = s->use_variance_statistic? 1: 0;
  //writer->headerPS.statistics[absoluteMax] = s->use_abs_max_statistic? 1: 0;

  writer->headerPS.ncuts=s->n_cuts;
  writer->lenPS=0;
  for (int i=0; i<s->n_cuts; i++) {
    writer->headerPS.nu_min[i]=s->nu_min[i];
    writer->headerPS.nu_max[i]=s->nu_max[i];
    writer->headerPS.fft_avg[i]=s->fft_avg[i];
    writer->headerPS.pssize[i]=s->pssize[i];
    assert(s->channel_mask==3);
    writer->lenPS+=s->pssize[i]*(4+12*(s->card_mask==3));

  }
  writer->tone_freq = 0;
  writer->lj_voltage0 = 0;
  writer->lj_diode = 0;
  printf ("Record size: %i\n", writer->lenPS);
  printf ("Version: %i\n", writer->headerPS.version);


  writer->average_recs=s->average_recs;
  writer->psbuftick = (float*)malloc(sizeof(float)*writer->lenPS * writer->average_recs);
  writer->psbuftock = (float*)malloc(sizeof(float)*writer->lenPS * writer->average_recs);
  writer->cleanps = (float*)malloc(sizeof(float)*writer->lenPS);
  writer->badps = (float*) malloc(sizeof(float)*writer->lenPS);
  writer->numbad = (int*) malloc(sizeof(int)*writer->lenPS);
  writer->rfi_sigma=s->n_sigma_null;
  writer->headerRFI.nSigma=s->n_sigma_null;
}

void resetAverage (WRITER *writer) {
  writer->writing=false;
  writer->crec=0;
  writer->fbad=0.0;
  writer->counter = 0;
  writer->totick=true;
}

void enableWriter(WRITER *wr, SETTINGS *set) {
  if (!wr->enabled) {
    wr->enabled=true;
    resetAverage(wr);
    maybeReOpenFile(wr, set, true);
  }
}

void disableWriter(WRITER *wr) {
  if (wr->enabled) {
    wr->enabled=false;
    if (wr->savethread.joinable()) wr->savethread.join();
    closeAndRename(wr);
  }
}


double getMJDNow()
{
  long int t=time(NULL);
  return (double)(t) / 86400.0  + 40587.0;
}



void processThread (WRITER& wrr, SETTINGS& setr) {
  size_t N=wrr.lenPS;
  int M=wrr.average_recs;
  wrr.writing=true;
  // note we process THE OTHER ONE
  float* ptr = wrr.totick ? wrr.psbuftock : wrr.psbuftick;
  for (size_t i=0;i<N;i++) {
    rfimean(&(ptr[i*M]),M,wrr.rfi_sigma, &(wrr.cleanps[i]), &(wrr.badps[i]), &(wrr.numbad[i]));
  }
  writerWritePS(&wrr,wrr.cleanps, &setr);
  int totbad=0;
  for (size_t i=0;i<N;i++) totbad+=wrr.numbad[i];
  writerWriteRFI(&wrr,wrr.badps,wrr.numbad,totbad);
  wrr.fbad=float(totbad)/(N*M);
  wrr.writing=false;
}


void writerAccumulatePS (WRITER *writer, float* ps, TWRITER *twr, SETTINGS *set) {
  if (writer->enabled) {
  if (writer->average_recs<=1) {
    writerWritePS(writer,ps,set);
    return;
  }
  size_t N=writer->lenPS;
  int M=writer->average_recs;
  size_t j=writer->crec;
  float* ptr = writer->totick ? writer->psbuftick : writer->psbuftock;
  for (size_t i=0;i<N;i++,j+=M) {
    ptr[j]=ps[i];
  }

  writer->crec++;
  if (writer->crec==M) {
    if (writer->savethread.joinable()) writer->savethread.join();
    writer->crec=0;
    writer->totick = not writer->totick;
    writer->savethread = std::thread(processThread,std::ref(*writer), std::ref(*set));
  }
  
  tprintfn(twr,1,"Writer Accumulator: %03d   Writing:%01d Tick/Tock:%01d  Reject in last save: %4.3f%%", 
	   writer->crec, writer->writing,writer->totick, writer->fbad*100);
  } else {
    tprintfn(twr,1,"Writer disabled.");
  }
}



void writerWritePS (WRITER *writer, float* ps, SETTINGS *set) {
  maybeReOpenFile(writer, set);
  double mjd=getMJDNow();
  fwrite (&mjd, sizeof(double), 1, writer->fPS);
  //fix here
  int numOutliersNulled=0;
  fwrite (&numOutliersNulled, sizeof(int), writer->headerPS.nChannels, writer->fPS);
  fwrite (ps, sizeof(float), writer->lenPS, writer->fPS);
  fwrite (&writer->tone_freq, sizeof(float), 1, writer->fPS);
  fwrite (&writer->lj_voltage0, sizeof(float), 1, writer->fPS);
  fwrite (&writer->lj_diode, sizeof(int), 1, writer->fPS);
  fflush(writer->fPS);
  writer->counter++;
}

void writerWriteRFI (WRITER *writer, float* ps, int* numbad, int totbad) {
  size_t N=writer->lenPS;
  int16_t totbads=totbad;
  // first  write the total number of bad records, for fat reading
  fwrite (&totbads,sizeof(int16_t), 1, writer->fRFI);
  for (int16_t i=0; i<N; i++) {
    if (numbad[i]>0) {
      fwrite (&i,sizeof(int16_t),1,writer->fRFI);
      int16_t tmp=numbad[i];
      fwrite (&tmp,sizeof(int16_t),1,writer->fRFI);
      fwrite (&ps[i],sizeof(float),1,writer->fRFI);
    }
  }
  fflush(writer->fRFI);
}



void writerCleanUp(WRITER *writer) {
  if (writer->savethread.joinable()) writer->savethread.join();
    if (writer->enabled) {
      closeAndRename(writer);
      printf ("Closing/renaming output files...\n");
    }
}
