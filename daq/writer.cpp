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
	fclose(writer->fps);
	fclose(writer->fout);
    }
    writer->counter =0; //reset sample counter to 0
    char afname[2][MAXFNLEN];
    sprintf(afname[0],writer->fname[0], ti->tm_year - 100 , ti->tm_mon + 1, 
	    ti->tm_mday, ti->tm_hour, ti->tm_min);
    sprintf(afname[1],writer->fname[1], ti->tm_year - 100 , ti->tm_mon + 1, 
	    ti->tm_mday, ti->tm_hour, ti->tm_min);
    printf ("New File: %s\n", afname);
    writer->fps=fopen(afname[0],"wb");
    writer->fout=fopen(afname[1],"wb");
    if (writer->fps==NULL) {
      printf ("CANNOT OPEN FILE:%s",afname[0]);
      exit(1);
    }
    if (writer->fout==NULL) {
      printf ("CANNOT OPEN FILE:%s",afname[1]);
      exit(1);
    }
    fwrite(&writer->header, sizeof(BMXHEADER),1, writer->fps);
    writer->reopen=false;
  }
  if (ti->tm_min%writer->save_every==1) {
      writer->reopen=true;
  }
}

void writerInit(WRITER *writer, SETTINGS *s) {
  printf ("==========================\n");
  strcpy(writer->fname[0],s->ps_output_pattern);
  strcpy(writer->fname[1],s->outlier_output_pattern);
  writer->save_every=s->save_every;
  writer->header.nChannels=1+s->channel_mask;
  writer->header.sample_rate=s->sample_rate;
  writer->header.fft_size=s->fft_size;
  writer->header.ncuts=s->n_cuts;
  writer->pslen=0.0;
  for (int i=0; i<s->n_cuts; i++) {
    writer->header.nu_min[i]=s->nu_min[i];
    writer->header.nu_max[i]=s->nu_max[i];
    writer->header.fft_avg[i]=s->fft_avg[i];
    writer->header.pssize[i]=s->pssize[i];
    writer->pslen+=s->pssize[i]*(1+3*(s->channel_mask==3));

  }
  writer->outlen = pow(2,s->log_chunk_size);
  writer->counter =0;
  printf ("Record size: %i\n", writer->pslen);
  printf ("Version: %i\n", writer->header.version);
  
  maybeReOpenFile(writer,true);
}

void writerWritePS (WRITER *writer, float* ps) {
  maybeReOpenFile(writer);
  fwrite (ps, sizeof(float), writer->pslen, writer->fps);
  fwrite (&writer->tone_freq, sizeof(float), 1, writer->fps);
  fflush(writer->fps);
  writer->counter++;
}

void writerWriteOutlier(WRITER * writer, int8_t * outlier, int chunk, int channel){
  maybeReOpenFile(writer);
  char * tag = (char *)malloc(100*sizeof(char));
  int tagLen=sprintf(tag, "sample: %d, chunk: %d, channel: %d", writer->counter, chunk, channel);
  fwrite(tag, sizeof(char), tagLen, writer->fout);
  fwrite (outlier, sizeof(int8_t), writer->pslen, writer->fout);
  fflush(writer->fout);
}

void writerCleanUp(WRITER *writer) {
  fclose(writer->fps);
  fclose(writer->fout);
}




