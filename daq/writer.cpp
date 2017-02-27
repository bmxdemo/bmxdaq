#include "writer.h"
#include "string.h"
#include "time.h"

void maybeReOpenFile(WRITER *writer, bool first=false) {
  time_t rawtime;   
  time ( &rawtime );
  struct tm *ti = localtime ( &rawtime );
  
  if (first || ((ti->tm_min%writer->save_every==0) && writer->reopen)) {
    if (!first) fclose(writer->f);
    char afname[MAXFNLEN];
    sprintf(afname,writer->fname, ti->tm_year - 100 , ti->tm_mon + 1, 
	    ti->tm_mday, ti->tm_hour, ti->tm_min);
    printf ("New File: %s\n", afname);
    writer->f=fopen(afname,"wb");
    fwrite(&writer->header, sizeof(BMXHEADER),1,writer->f);
    writer->reopen=false;
  }
  if (ti->tm_min%writer->save_every==1) writer->reopen=true;
}

void writerInit(WRITER *writer, SETTINGS *s) {
  printf ("\n\nInitializing writer\n");
  printf ("==========================\n");
  strcpy(writer->fname,s->output_pattern);
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
  printf ("Record size: %i\n", writer->pslen);
  printf ("Version: %i\n", writer->header.version);

  maybeReOpenFile(writer,true);
}

void writerWritePS (WRITER *writer, float* ps) {
  maybeReOpenFile(writer);
  fwrite (ps, sizeof(float), writer->pslen, writer->f);
  fwrite (&writer->tone_freq, sizeof(float), 1, writer->f);
  fflush(writer->f);
}

void writerCleanUp(WRITER *writer) {
  fclose(writer->f);
}
