#pragma once

#include "stdint.h"

#define MAXCHAR 512
#define MAXCUTS 10
#define MAXFREQ 10

// modifiable settings
struct SETTINGS {
  // basic settings

  // digi card settings
  float sample_rate; // in samples/s
  long long int spc_sample_rate;
  long long int spc_ref_clock;
  uint64_t channel_mask;  // channel bit mask 
  int32_t ADC_range; // in mV
  int ext_clock_mode; // 0 for internal, 1 for external

  // simulate card
  int simulate_digitizer;
  // dont process, just transfer from digitizer
  int dont_process;
  
  // number of samples that we want to take, zero for forver
  long int nsamples;

  // waveform file and length, if zero, don't save
  long int wave_nbytes;
  char wave_fname[MAXCHAR];
  

  // size of FFT transform
  uint32_t fft_size; // must be power of 2
  int n_cuts;
  float nu_min[MAXCUTS], nu_max[MAXCUTS]; // min and max frequency to output
  uint32_t fft_avg[MAXCUTS]; // how many bins to average over
  int buf_mult; // buffer multiplier, we allocate
                //buf_mult*fft_size for transfer
  //
  int cuda_streams; // number of cuda streams
  int cuda_threads; // number of cuda threads

  // output options
  char ps_output_pattern[MAXCHAR];
  char rfi_output_pattern[MAXCHAR];
  int save_every;
  
  // printout options
  int print_meanvar;
  int print_maxp;
  
  // "derived" quantities for passing
  int pssize[MAXCUTS];

  // frequency generator
  int fg_nfreq;
  int fg_baudrate;
  int fg_switchevery;
  char fg_port[MAXCHAR];
  float fg_freq[MAXFREQ];
  float fg_ampl[MAXFREQ];


  //RFI rejection
  int log_chunk_size; //log base 2 of chunk size to be used to collect RFI statistics
  int n_sigma; //number of standard deviations used to determine outliers.
};

// Fixed defines

#define VERSION "0.01"


void init_settings(SETTINGS *settings, char* inifile);
void print_settings(SETTINGS *s);
