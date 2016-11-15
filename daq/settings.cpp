#include "settings.h"
#include "stdio.h"
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

int my_linecount(FILE *f)
{
  int i0=0;
  char ch[1000];
  while((fgets(ch,sizeof(ch),f))!=NULL) {
    i0++;
  }
  return i0;
}

void init_settings(SETTINGS *s, char* fname) {
    s->sample_rate=1.25e9;
    s->fft_size = (1<<26);
    s->nu_min=0;
    s->nu_max=(float)(s->sample_rate/2);
    s->fft_avg=16384;
    s->channel_mask=3; // which channels, both to start with
    s->ADC_range=1000;
    s->ext_clock_mode=0;
    s->buf_mult=8;
    s->cuda_streams=1;
    s->cuda_threads=1024;
    s->simulate_digitizer=0;
    s->dont_process=0;
    s->save_every=60;
    sprintf(s->output_pattern,"%02d%02d%02d_%02d%02d.data");
    if (fname) {
         FILE *fi;
	 int n_lin,ii;
	 //Read parameters from file
	 fi=fopen(fname,"r");
	 if (!fi) {
	   printf ("Error opening %s\n",fname);
	   exit(1);
	 }
	 n_lin=my_linecount(fi); rewind(fi);
	 for(ii=0;ii<n_lin;ii++) {
	   char s0[512],s1[64],s2[256];
	   if(fgets(s0,sizeof(s0),fi)==NULL) {
	     printf("Error reading line %d, file %s\n",ii+1,fname);
	     exit(1);
	   }
	   if((s0[0]=='#')||(s0[0]=='\n')||(s0[0]==' ')) continue;
	   int sr=sscanf(s0,"%s %s",s1,s2);
	   if(sr!=2) {
	     printf("Error reading line %d, file %s\n",ii+1,fname);
	     exit(1);
	   }
	   if(!strcmp(s1,"sample_rate="))
	     s->sample_rate=atof(s2)*1e6;
	   else if(!strcmp(s1,"FFT_power="))
	     s->fft_size = ( 1 << atoi(s2));
	   else if(!strcmp(s1,"buf_mult="))
	     s->buf_mult = ( 1 << atoi(s2));
	   else if(!strcmp(s1,"nu_min="))
	     s->nu_min=atof(s2)*1e6;
	   else if(!strcmp(s1,"nu_max=")) {
	     if (atof(s2)>0) 
	       s->nu_max=atof(s2)*1e6;
	   }  else if(!strcmp(s1,"channel_mask="))
	     s->channel_mask=atoi(s2);
	   else if(!strcmp(s1,"ADC_range="))
	     s->ADC_range=atoi(s2);
	   else if(!strcmp(s1,"ext_clock_mode="))
	     s->ext_clock_mode=atoi(s2);
	   else if(!strcmp(s1,"cuda_streams="))
	     s->cuda_streams=atoi(s2);
	   else if(!strcmp(s1,"cuda_threads="))
	     s->cuda_threads=atoi(s2);
	   else if(!strcmp(s1,"simulate_digitizer="))
	     s->simulate_digitizer=atoi(s2);
	   else if(!strcmp(s1,"dont_process="))
	     s->dont_process=atoi(s2);
	   else if(!strcmp(s1,"fft_avg="))
	     s->fft_avg=atoi(s2);
	   else if(!strcmp(s1,"save_every="))
	     s->save_every=atoi(s2);
	   else if(!strcmp(s1,"output_pattern="))
	     strcpy(s->output_pattern,s2);
	   else {
	     printf("Unknown parameter %s\n",s1);
	     exit(1);
	   }
	 }
	 fclose(fi);
     }
}

void print_settings(SETTINGS *s) {
  printf ("\n******************************************************************\n\n");
  printf ("BMX DAQ, version %s \n\n",VERSION);
  printf ("Sampling rate: %5.3g GS\n", s->sample_rate/1e9);
  printf ("FFT buffer size: %i\n", s->fft_size);
  printf ("Notify size: %iMB\n", s->fft_size*(1+(s->channel_mask==3))/(1024*1024));
  printf ("FFT buffer size in ms: %5.3g \n", s->fft_size/s->sample_rate*1000.);
  printf ("Simulate digitizer: %i \n", s->simulate_digitizer);
  printf ("Nu min: %5.3g MHz\n", s->nu_min/1e6);
  printf ("Nu max: %5.3g MHz\n", s->nu_max/1e6);
  printf ("FFT avg block: %i\n", s->fft_avg);
  printf ("Full number of PS bins: %i\n",s->fft_size/2/s->fft_avg);
  printf ("Channel mask: %lu\n", s->channel_mask);
  printf ("ADC range: %imV\n", s->ADC_range);
  printf ("External clock mode: %i\n", s->ext_clock_mode);
  printf ("GPU CUDA streams: %i\n", s->cuda_streams);
  printf ("GPU CUDA threads: %i\n", s->cuda_threads);
  printf ("Buffer multiplier (size of ADC buffer in FFT buf size): %i\n", s->buf_mult);
  printf ("\n*********************************************************************\n");
}
