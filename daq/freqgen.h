#pragma once


/*
**************************************************************************
bDoCardSetuo: setup matching the calculation routine
**************************************************************************
*/

#include "settings.h"
#include "writer.h"

struct FREQGEN {
  int   cport; //port number
  int   bdrate; // baud rate
  int   ccount, cswitch;  // how often we switch
  int cfreq, nfreq; // frequency list counter
  float freq[MAXFREQ]; // frequencies in MHz
  float ampl[MAXFREQ]; // amplitudes in Vpp
};


//initialize
void freqGenInit (FREQGEN *fg, WRITER* wr, SETTINGS *set);

//main worker loop
void freqGenLoop (FREQGEN *fg, WRITER* wr);

//shutdown
void freqGenCleanUp(FREQGEN *fg);
