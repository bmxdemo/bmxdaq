#pragma once
#include "settings.h"
#include "rfi.h"
#include "stdio.h"
#include <stdint.h>
// character lengths
#define MAXFNLEN 512
// version of BMXHEADER structure to implement
// in python readers, etc.
// CHANGES:
//     v2 -- save float with cur tone freq every time you save
//     v3 -- save MJD double
//     v4 -- save labjack voltage float and diode
//     v5 -- changed localtime to gmtime in filename -- header hasn't changed
//     v6 -- two digitizers (not implemented yet, FIX -- AS)
#define HEADERVERSION 6


struct BMXHEADER {
  const char magic[8]=">>BMX<<"; // magic header char to recogize files *BMX*
  int version=HEADERVERSION;
  int nChannels;
  float sample_rate;
  uint32_t fft_size; 
  int32_t ADC_range;
  bool  statistics[STAT_COUNT_MINUS_ONE + 1]; //array indicating which statistics are being used
  int ncuts;
  float nu_min[MAXCUTS], nu_max[MAXCUTS];
  uint32_t fft_avg[MAXCUTS];
  int pssize[MAXCUTS];
};

struct RFIHEADER {
  const char magic[8]=">>RFI<<";
  int chunkSize; //number of elements per chunk 
  float nSigma;    //number of sigma away from mean
};

struct WRITER {
  char fnamePS[MAXFNLEN], fnameRFI[MAXFNLEN], fnameLastBuffer[MAXFNLEN];  //file names, from settings
  char afnamePS[MAXFNLEN], afnameRFI[MAXFNLEN],  afnameLastBuffer[MAXFNLEN];  //current file names
  char tafnamePS[MAXFNLEN], tafnameRFI[MAXFNLEN];  //temporary current file names 
                                                   //(with ".new")
  uint32_t lenPS; // full length of PS info
  uint32_t lenRFI; //length of outlier chunk
  int save_every; // how many minutes we save.
  FILE* fPS, *fRFI;
  bool reopen;
  BMXHEADER headerPS;  //header for power spectra files
  RFIHEADER headerRFI; //header for rfi files
  float tone_freq;
  float lj_voltage0;
  int lj_diode;
  int counter; //number of PS written to current file
};



void writerInit(WRITER *writer, SETTINGS *set, bool isRFIOn);
void writerWritePS (WRITER *writer, float* ps, int * numOutliersNulled, bool isRFIOn);
void writerWriteRFI(WRITER *writer, int8_t * outlier, int chunk, int channel, float* nSigma);
void writerWriteLastBuffer(WRITER *writer, int8_t ** bufstart, int numCards, int size);
void writerCleanUp(WRITER *writer, bool isRFIOn);

void closeAndRename(WRITER *writer);

