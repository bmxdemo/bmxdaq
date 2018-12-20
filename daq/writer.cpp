#include "writer.h"
#include "string.h"
#include "time.h"
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

void maybeReOpenFile(WRITER *writer, bool first=false) {
  time_t rawtime;   
  time ( &rawtime );
  struct tm *ti = gmtime ( &rawtime );
  
  if (first || ((ti->tm_min%writer->save_every==0) && writer->reopen)) {
    if (!first) closeAndRename(writer);

    writer->counter =0; //reset sample counter to 0
    sprintf(writer->afnamePS,writer->fnamePS, ti->tm_year - 100 , ti->tm_mon + 1, 
	    ti->tm_mday, ti->tm_hour, ti->tm_min);
    sprintf(writer->tafnamePS,"%s.new",writer->afnamePS);
    printf ("New File: %s\n", writer->tafnamePS);
    writer->fPS=fopen(writer->tafnamePS,"wb");
    
    if (writer->fPS==NULL) {
      printf ("CANNOT OPEN FILE:%s",writer->tafnamePS);
      exit(1);
    }
    
    fwrite(&writer->headerPS, sizeof(BMXHEADER),1, writer->fPS);

    if(writer->rfiOn){
      sprintf(writer->afnameRFI,writer->fnameRFI, ti->tm_year - 100 , ti->tm_mon + 1, 
	      ti->tm_mday, ti->tm_hour, ti->tm_min);
      sprintf(writer->tafnameRFI,"%s.new",writer->afnameRFI);
      printf ("New File: %s\n", writer->tafnameRFI);
      writer->fRFI=fopen(writer->tafnameRFI,"wb");
      if (writer->fRFI==NULL) {
        printf ("CANNOT OPEN FILE:%s",writer->tafnameRFI);
        exit(1);
      }
      fwrite(&writer->headerRFI, sizeof(RFIHEADER), 1, writer->fRFI);    
    }
    writer->reopen=false;
  }

  if (ti->tm_min%writer->save_every==1)  writer->reopen=true;
  
}

void writerInit(WRITER *writer, SETTINGS *s) {
  printf ("==========================\n");
  strcpy(writer->fnamePS,s->ps_output_pattern);
  strcpy(writer->fnameRFI,s->rfi_output_pattern);
  writer->save_every=s->save_every;
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
  writer->lenPS=0.0;
  for (int i=0; i<s->n_cuts; i++) {
    writer->headerPS.nu_min[i]=s->nu_min[i];
    writer->headerPS.nu_max[i]=s->nu_max[i];
    writer->headerPS.fft_avg[i]=s->fft_avg[i];
    writer->headerPS.pssize[i]=s->pssize[i];
    assert(s->channel_mask==3);
    writer->lenPS+=s->pssize[i]*(4+12*(s->card_mask==3));

  }
  writer->counter = 0;
  writer->tone_freq = 0;
  writer->lj_voltage0 = 0;
  writer->lj_diode = 0;
  printf ("Record size: %i\n", writer->lenPS);
  printf ("Version: %i\n", writer->headerPS.version);

  maybeReOpenFile(writer, true);
}

double getMJDNow()
{
  long int t=time(NULL);
  return (double)(t) / 86400.0  + 40587.0;
}

void writerWritePS (WRITER *writer, float* ps) {
  maybeReOpenFile(writer);
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


// void writerWriteRFI(WRITER * writer, int8_t * outlier, int chunk, int channel, float *nSigma){
//   maybeReOpenFile(writer, true);
//   fwrite(&writer->counter, sizeof(int), 1, writer->fRFI);
//   fwrite(&chunk, sizeof(int), 1, writer->fRFI);
//   fwrite(&channel, sizeof(int), 1, writer->fRFI);
//   fwrite(nSigma, sizeof(float), STAT_COUNT_MINUS_ONE +1, writer->fRFI);

//   fwrite (outlier, sizeof(int8_t), writer->lenRFI, writer->fRFI);
//   fflush(writer->fRFI);
// }

// void writerWriteLastBuffer(WRITER * writer, int8_t ** bufstart, int numCards, int size){
//   time_t rawtime;
//   time (&rawtime);
//   struct tm *ti = localtime ( &rawtime );
//   sprintf(writer->afnameLastBuffer, writer->fnameLastBuffer, ti->tm_year - 100 , ti->tm_mon + 1,
//             ti->tm_mday, ti->tm_hour, ti->tm_min);
//   printf("Creating: %s\n", writer->afnameLastBuffer);
//   FILE * fw = fopen(writer->afnameLastBuffer, "wb");
//   if(fw == NULL)
// 	printf("CANNOT OPEN FILE: %s\n", writer->afnameLastBuffer);
//   else {
//     for(int i=0; i < numCards; i++)
//   	  fwrite(bufstart[i], sizeof(int8_t), size, fw);
//   	fclose(fw);
//   }
// }

void writerCleanUp(WRITER *writer) {
  printf ("Closing/renaming output files...\n");
  closeAndRename(writer);
}
