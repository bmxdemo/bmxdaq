#pragma once

#include "stdint.h"

#define MAXCHAR 512
#define MAXCUTS 10
#define MAXFREQ 10
#define MAXSTREAMS 8
#define NUM_DELAYS 200
#define NDELAYBUFS 4

// modifiable settings
struct SETTINGS {
  // debug?
  int debug;
  // daq num. 1 is captain, 2 is sailor
  int daqNum;
  
  // basic settings
  uint64_t card_mask; //bit mask representing which cards are to be used
  // digi card settings
  char  card1[MAXCHAR], card2[MAXCHAR];
  
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
  int cuda_streams; // number of cuda streams
  int cuda_threads; // number of cuda threads
  
  // ring buffer
  int ringbuffer_size;
  // ring buffer force memcpy
  int ringbuffer_force;

  // delay calibration

  unsigned short int bufdelay[2]; // digital delays in full buffers 
  unsigned int delay[2]; // digital deltays for card 1,2 in samples on top of bufdelay
  int measure_delay; // measure delays between cards 1 and 2
  

  // output options
  char ps_output_pattern[MAXCHAR];
  char rfi_output_pattern[MAXCHAR];
  char ringbuffer_output_pattern[MAXCHAR];
  int new_file_every;
  int average_recs; // how many records to average

  // printout options
  int print_meanvar;
  int print_maxp;
  int print_every;

  // "derived" quantities for passing
  int pssize[MAXCUTS];

  // frequency generator
  int fg_nfreq;
  int fg_baudrate;
  int fg_switchevery;
  char fg_port[MAXCHAR];
  float fg_freq[MAXFREQ];
  float fg_ampl[MAXFREQ];


  // labjack
  int lj_Noff; // number of samples with diode off
  int lj_Non;  // number of samples with diode on;

  //RFI rejection
  float n_sigma_null; //number of standard deviations used to determine outliers to null out. 0 for none
};

// Fixed defines

#define VERSION "1.0_multi"


void init_settings(SETTINGS *settings, const char* inifile);
void print_settings(SETTINGS *s);
