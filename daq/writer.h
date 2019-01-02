#pragma once
#include "terminal.h"
#include "settings.h"
#include "stdio.h"
#include <stdint.h>
#include <thread>


// character lengths
#define MAXFNLEN 512
// version of BMXHEADER structure to implement
// in python readers, etc.
// CHANGES:
//     v2 -- save float with cur tone freq every time you save
//     v3 -- save MJD double
//     v4 -- save labjack voltage float and diode
//     v5 -- changed localtime to gmtime in filename -- header hasn't changed
//     v6 -- two digitizers 
#define HEADERVERSION 6


struct BMXHEADER {
  const char magic[8]=">>BMX<<"; // magic header char to recogize files *BMX*
  int version=HEADERVERSION;
  int cardMask;
  int nChannels;
  float sample_rate;
  uint32_t fft_size; 
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
  bool enabled;
  char fnamePS[MAXFNLEN], fnameRFI[MAXFNLEN]; //file names, from settings
  char afnamePS[MAXFNLEN], afnameRFI[MAXFNLEN];  //current file names
  char tafnamePS[MAXFNLEN], tafnameRFI[MAXFNLEN];  //temporary current file names //(with ".new")
  bool rfiOn;
  uint32_t lenPS; // full length of PS info
  uint32_t lenRFI; //length of outlier chunk
  int new_file_every; // how many minutes we save.
  int average_recs; // how many records to average over
  int rfi_sigma;
  int crec;
  bool totick, writing;
  std::thread savethread;
  float *psbuftick, *psbuftock, *cleanps, *badps;
  int *numbad;
  float fbad; //fraction bad last time
  
 
  FILE* fPS, *fRFI;
  bool reopen;
  BMXHEADER headerPS;  //header for power spectra files
  RFIHEADER headerRFI; //header for rfi files
  float tone_freq;
  float lj_voltage0;
  int lj_diode;
  int counter; //number of PS written to current file
};



void writerInit(WRITER *writer, SETTINGS *set);

float rfimean (float arr[], int n, int nsigma, float *cleanmean, float *outliermean, int *numbad);

void writerWritePS (WRITER *writer, float* ps);

void writerAccumulatePS (WRITER *writer, float* ps, TWRITER *twr);

void enableWriter(WRITER *wr);
void disableWriter(WRITER *wr);



//void writerWriteRFI(WRITER *writer, int8_t * outlier, int chunk, int channel, float* nSigma);
//void writerWriteLastBuffer(WRITER *writer, int8_t ** bufstart, int numCards, int size);
void writerCleanUp(WRITER *writer);

void closeAndRename(WRITER *writer);

