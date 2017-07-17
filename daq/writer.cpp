#include "writer.h"
#include "string.h"
#include "time.h"
#include "stdlib.h"
#include <math.h>

void maybeReOpenFile(WRITER *writer, bool first=false) {
  time_t rawtime;   
  time ( &rawtime );
  struct tm *ti = localtime ( &rawtime );
  
  if (first || ((ti->tm_min%writer->save_every==0) && writer->reopen)) {
    if (!first){
	fclose(writer->fPS);
	fclose(writer->fRFI);
    }
    writer->counter =0; //reset sample counter to 0
    char afnamePS[MAXFNLEN], afnameRFI[MAXFNLEN]; //file names
    sprintf(afnamePS,writer->fnamePS, ti->tm_year - 100 , ti->tm_mon + 1, 
	    ti->tm_mday, ti->tm_hour, ti->tm_min);
    sprintf(afnameRFI,writer->fnameRFI, ti->tm_year - 100 , ti->tm_mon + 1, 
	    ti->tm_mday, ti->tm_hour, ti->tm_min);
    printf ("New File: %s\n", afnamePS);
    printf ("New File: %s\n", afnameRFI);
    writer->fPS=fopen(afnamePS,"wb");
    writer->fRFI=fopen(afnameRFI,"wb");
    
    if (writer->fPS==NULL) {
      printf ("CANNOT OPEN FILE:%s",afnamePS);
      exit(1);
    }
    if (writer->fRFI==NULL) {
      printf ("CANNOT OPEN FILE:%s",afnameRFI);
      exit(1);
    }
    
    fwrite(&writer->headerPS, sizeof(BMXHEADER),1, writer->fPS);
    fwrite(&writer->headerRFI, sizeof(RFIHEADER), 1, writer->fRFI);

    writer->reopen=false;
  }
  if (ti->tm_min%writer->save_every==1) {
      writer->reopen=true;
  }
}

void writerInit(WRITER *writer, SETTINGS *s) {
  printf ("==========================\n");
  strcpy(writer->fnamePS,s->ps_output_pattern);
  strcpy(writer->fnameRFI,s->rfi_output_pattern);
  writer->save_every=s->save_every;
  writer->headerPS.nChannels=1+s->channel_mask;
  writer->headerPS.sample_rate=s->sample_rate;
  writer->headerPS.fft_size=s->fft_size;
  writer->headerPS.ncuts=s->n_cuts;
  writer->headerRFI.chunkSize = pow(2, s->log_chunk_size);
  writer->headerRFI.nSigma = s->n_sigma;
  writer->lenPS=0.0;
  for (int i=0; i<s->n_cuts; i++) {
    writer->headerPS.nu_min[i]=s->nu_min[i];
    writer->headerPS.nu_max[i]=s->nu_max[i];
    writer->headerPS.fft_avg[i]=s->fft_avg[i];
    writer->headerPS.pssize[i]=s->pssize[i];
    writer->lenPS+=s->pssize[i]*(1+3*(s->channel_mask==3));

  }
  writer->lenRFI = pow(2,s->log_chunk_size);
  writer->counter =0;
  printf ("Record size: %i\n", writer->lenPS);
  printf ("Version: %i\n", writer->headerPS.version);
  
  maybeReOpenFile(writer,true);
}

void writerWritePS (WRITER *writer, float* ps) {
  maybeReOpenFile(writer);
  fwrite (ps, sizeof(float), writer->lenPS, writer->fPS);
  fwrite (&writer->tone_freq, sizeof(float), 1, writer->fPS);
  fflush(writer->fPS);
  writer->counter++;
}

void writerWriteRFI(WRITER * writer, int8_t * outlier, int chunk, int channel){
  maybeReOpenFile(writer);
  fwrite(&writer->counter, sizeof(int), 1, writer->fRFI);
  fwrite(&chunk, sizeof(int), 1, writer->fRFI);
  fwrite(&channel, sizeof(int), 1, writer->fRFI);
  fwrite (outlier, sizeof(int8_t), writer->lenRFI, writer->fRFI);
  fflush(writer->fRFI);
}

void writerCleanUp(WRITER *writer) {
  fclose(writer->fPS);
  fclose(writer->fRFI);
}




