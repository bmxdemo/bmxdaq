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
    s->spc_sample_rate=1250*1000000;
    s->spc_ref_clock=1250*1000000;
    s->fft_size = (1<<26);
    s->n_cuts=1;
    s->nu_min[0]=0;
    s->nu_max[0]=(float)(s->sample_rate/2);
    s->fft_avg[0]=16384;
    s->channel_mask=3; // which channels, both to start with
    s->ADC_range=1000;
    s->ext_clock_mode=0;
    s->buf_mult=8;
    s->cuda_streams=2;
    s->cuda_threads=1024;
    s->simulate_digitizer=1;
    s->dont_process=0;
    s->save_every=60;
    s->print_meanvar=1;
    s->print_maxp=0;
    s->print_every=1;
    sprintf(s->ps_output_pattern,"%%02d%%02d%%02d_%%02d%%02d.data");
    sprintf(s->rfi_output_pattern,"%%02d%%02d%%02d_%%02d%%02d.outliers");
    s->fg_nfreq=0;
    s->fg_baudrate=9600;
    s->fg_switchevery=10;
    sprintf(s->fg_port,"ttyS0");
    s->log_chunk_size = 20;
    s->n_sigma_null = 0;
    s->n_sigma_write = 0;
    s->use_mean_statistic = false;
    s->use_variance_statistic = false;
    s->use_abs_max_statistic = false;
    
    s->nsamples=0;
    s->wave_nbytes=0;
    sprintf(s->wave_fname,"wave.bin");
     
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

	   bool found=true;
	   if(!strcmp(s1,"sample_rate="))
	     s->sample_rate=atof(s2)*1e6;
	   else if(!strcmp(s1,"spc_sample_rate="))
	     s->spc_sample_rate=atoi(s2)*1000000;
	   else if(!strcmp(s1,"spc_ref_clock="))
	     s->spc_ref_clock=atoi(s2)*1000000;
	   else if(!strcmp(s1,"FFT_power="))
	     s->fft_size = ( 1 << atoi(s2));
	   else if(!strcmp(s1,"buf_mult="))
	     s->buf_mult = ( 1 << atoi(s2));
	   else if(!strcmp(s1,"channel_mask="))
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
	   else if(!strcmp(s1,"n_cuts="))
	     s->n_cuts=atoi(s2);
	   else if(!strcmp(s1,"save_every="))
	     s->save_every=atoi(s2);
	   else if(!strcmp(s1,"ps_output_pattern="))
	     strcpy(s->ps_output_pattern,s2);
	   else if(!strcmp(s1,"rfi_output_pattern="))
	     strcpy(s->rfi_output_pattern,s2);
	   else if(!strcmp(s1,"print_meanvar="))
	     s->print_meanvar=atoi(s2);
	   else if(!strcmp(s1,"print_maxp="))
	     s->print_maxp=atoi(s2);
	   else if(!strcmp(s1,"print_every="))
	     s->print_every=atoi(s2);
	   else if(!strcmp(s1,"fg_nfreq="))
	     s->fg_nfreq=atoi(s2);
	   else if(!strcmp(s1,"fg_baudrate="))
	     s->fg_baudrate=atoi(s2);
	   else if(!strcmp(s1,"fg_switchevery="))
	     s->fg_switchevery=atoi(s2);
	   else if(!strcmp(s1,"fg_port="))
	     strcpy(s->fg_port,s2);
	   else if(!strcmp(s1,"wave_fname="))
	     strcpy(s->wave_fname,s2);
	   else if(!strcmp(s1,"wave_nbytes="))
	     s->wave_nbytes=atoi(s2);
	   else if(!strcmp(s1,"log_chunk_size="))
	     s->log_chunk_size=atoi(s2);
	   else if(!strcmp(s1,"n_sigma_null="))
	     s->n_sigma_null=atoi(s2);
	   else if(!strcmp(s1,"n_sigma_write="))
	     s->n_sigma_write=atoi(s2);
	   else if(!strcmp(s1,"use_mean_statistic="))
	     s->use_mean_statistic=atoi(s2);
	   else if(!strcmp(s1,"use_variance_statistic="))
	     s->use_variance_statistic=atoi(s2);
	   else if(!strcmp(s1,"use_abs_max_statistic="))
	     s->use_abs_max_statistic=atoi(s2);
	   else if(!strcmp(s1,"nsamples="))
	     s->nsamples=atoi(s2);
	   else found=false;

	   if (!found) {
	     for (int i=0;i<MAXCUTS;i++) {
	       char tmpstr[MAXCHAR];
	       sprintf(tmpstr, "nu_min%i=",i);
	       if(!strcmp(s1,tmpstr)) {
		 found=true;
		 s->nu_min[i]=atof(s2)*1e6;
		 break;
	       }
	       sprintf(tmpstr, "nu_max%i=",i);
	       if(!strcmp(s1,tmpstr)) {
		 double tmp;
		 if (atof(s2)>0)
		   tmp=atof(s2)*1e6;
		 else
		   tmp=(float)(s->sample_rate/2);
		 s->nu_max[i]=tmp;
		 found=true;
		 break;
	       }
	       sprintf(tmpstr, "fft_avg%i=",i);
	       if(!strcmp(s1,tmpstr)) {
		 found=true;
		 s->fft_avg[i]=atoi(s2);
		 break;
	       }
	     }
	     for (int i=0;i<MAXFREQ;i++) {
	       char tmpstr[MAXCHAR];
	       sprintf(tmpstr, "fg_freq%i=",i);
	       if(!strcmp(s1,tmpstr)) {
		 found=true;
		 s->fg_freq[i]=atof(s2);
		 break;
	       }
	       sprintf(tmpstr, "fg_ampl%i=",i);
	       if(!strcmp(s1,tmpstr)) {
		 if (atof(s2)>0) 
		   s->fg_ampl[i]=atof(s2);
		 found=true;
		 break;
	       }
	     }
	   }
	   if (!found) {
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
  printf ("# PS cuts: %i \n", s->n_cuts);
  for (int i=0;i<s->n_cuts;i++) {
    printf ("  Nu min [%i]: %5.3g MHz\n", i, s->nu_min[i]/1e6);
    printf ("  Nu max [%i]: %5.3g MHz\n", i, s->nu_max[i]/1e6);
    printf ("  FFT avg block: [%i] %i\n", i, s->fft_avg[i]);
    // not yet initialized here
    //printf ("  Full number of PS bins [%i]: %i\n",i, s->fft_size/2/s->fft_avg[i]);
  }
  printf ("Channel mask: %lu\n", s->channel_mask);
  printf ("ADC range: %imV\n", s->ADC_range);
  printf ("External clock mode: %i\n", s->ext_clock_mode);
  printf ("GPU CUDA streams: %i\n", s->cuda_streams);
  printf ("GPU CUDA threads: %i\n", s->cuda_threads);
  printf ("Buffer multiplier (size of ADC buffer in FFT buf size): %i\n", s->buf_mult);
  printf ("\n*********************************************************************\n");
}
