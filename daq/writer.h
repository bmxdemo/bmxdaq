#pragma once
#include "settings.h"
#include "stdio.h"
#include "time.h"
#include <stdint.h>
// character lengths
#define MAXFNLEN 512
// version of BMXHEADER structure to implement
// in python readers, etc.
// CHANGES:
//     v2 -- save float with cur tone freq every time you save
#define HEADERVERSION 2


struct BMXHEADER {
  const char magic[8]=">>BMX<<"; // magic header char to recogize files *BMX*
  int version=HEADERVERSION;
  int nChannels;
  float sample_rate;
  uint32_t fft_size; 

  int ncuts;
  float nu_min[MAXCUTS], nu_max[MAXCUTS];
  uint32_t fft_avg[MAXCUTS];
  int pssize[MAXCUTS];
};


struct WRITER {
  char fname[2][MAXFNLEN];
  uint32_t pslen; // full length of PS info
  uint32_t outlen; //length of outlier chunks
  int save_every; // how many minutes we save.
  FILE* fps; //file for PS
  FILE* fout; //file for outliers
  bool reopen;
  BMXHEADER header;
  float tone_freq;
  int counter; //number of PS written to current file
};


void writerInit(WRITER *writer, SETTINGS *set);
void writerWritePS (WRITER *writer, float* ps);
void writerWriteOutlier(WRITER *writer, int8_t * outlier, int chunk, int channel);
void writerCleanUp(WRITER *writer);

