/*

The digitizer GPU driver.

Based on examples by Spectrum GmbH.

Anze Slosar, anze@bnl.gob

**************************************************************************
*/

#include "settings.h"
#include "digicard.h"
#include "gpucard.h"
#include "freqgen.h"

// ----- standard c include files -----
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_TERM_CHARACTERS = 1000 //maximum number of characters needed for terminal buffer

/*
**************************************************************************
main 
**************************************************************************
*/

int main(int argc,char **argv)
{ 
  char                szBuffer[1024];     // a character buffer for any messages
  SETTINGS settings;                      // settings
  WRITER writer;                          // writer module
  DIGICARD dcard;                         // digitizer CARD
  GPUCARD gcard;                          // GPU card
  FREQGEN fgen;

  if(argc>=2) {
    char fname_ini[256];
    sprintf(fname_ini,"%s",argv[1]);
    init_settings(&settings,fname_ini);
  } else
    init_settings(&settings,NULL);

  // intialize
  print_settings(&settings);
  digiCardInit(&dcard,&settings);
  if (!settings.dont_process) gpuCardInit(&gcard,&settings);
  if (settings.fg_nfreq) freqGenInit(&fgen, &writer, &settings);
  writerInit(&writer,&settings);
  
  //work
  digiWorkLoop(&dcard, &gcard, &settings, &fgen, &writer);
  //shutdown
  digiCardCleanUp(&dcard, &settings);
  writerCleanUp(&writer);
  if (settings.fg_nfreq) freqGenCleanUp(&fgen);
  return 0;
}

