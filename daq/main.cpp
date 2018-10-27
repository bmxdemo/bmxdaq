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
#include "terminal.h"
#include "ljack.h"

// ----- standard c include files -----
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define TERMINAL_BUFFER_SIZE 10000 //maximum size of terminal output

/*
**************************************************************************
main 
**************************************************************************
*/

int main(int argc,char **argv)
{ 
  char szBuffer[1024];                    // a character buffer for any messages
  SETTINGS settings;                      // settings
  WRITER writer;                          // writer module
  DIGICARD dcard;                         // digitizer CARD
  GPUCARD gcard;                          // GPU card
  FREQGEN fgen;                           // Freq generator
  LJACK ljack;                            // Labjack
  RFI rfi;                                //rfi stuff
  TWRITER twriter;			  //terminal writers

  if(argc>=2) {
    char fname_ini[256];
    sprintf(fname_ini,"%s",argv[1]);
    init_settings(&settings,fname_ini);
  } else
    init_settings(&settings,NULL);

  // intialize
  print_settings(&settings);
  if (settings.channel_mask!=3) {
    printf ("Sorry, the code is only really working for 2 channels.\n");
    return 1;
  }


  // first ephemeral things
  if (settings.fg_nfreq) freqGenInit(&fgen, &writer, &settings);
  if (settings.lj_Non) LJInit(&ljack, &writer, &settings);
  // digitizer
  digiCardInit(&dcard,&settings);

  // GPU
  if (!settings.dont_process) gpuCardInit(&gcard,&settings);
  // RFI
  rfiInit(&rfi, &settings, &gcard);
  // writer
  writerInit(&writer,&settings, rfi.isRFIOn);

  //MAIN LOOP
  digiWorkLoop(&dcard, &gcard, &settings, &fgen, &ljack, &writer, &twriter, &rfi);

  //shutdown
  digiCardCleanUp(&dcard, &settings);
  writerCleanUp(&writer, rfi.isRFIOn);
  if (settings.fg_nfreq) freqGenCleanUp(&fgen);
  if (settings.lj_Non) LJCleanUp(&ljack);

  printf ("Done.\n");
  return 0;
}

