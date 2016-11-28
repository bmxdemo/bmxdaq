#pragma once
#include "settings.h"
#include "stdio.h"
#include "time.h"
// character lengths
#define MAXFNLEN 512
// version of BMXHEADER structure to implement
// in python readers, etc.
#define HEADERVERSION 1 


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
  char fname[MAXFNLEN];
  int pslen; // full length of PS info
  int save_every; // how many minutes we save.
  FILE* f;
  bool reopen;
  BMXHEADER header;
};


void writerInit(WRITER *writer, SETTINGS *set);
void writerWritePS (WRITER *writer, float* ps);
void writerCleanUp(WRITER *writer);

