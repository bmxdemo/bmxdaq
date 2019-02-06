/*

The digitizer GPU driver.

Based on examples by Spectrum GmbH.

Anze Slosar, anze@bnl.gob

**************************************************************************
*/

#include "settings.h"
#include "digicard.h"
#include "UDPCommunication.h"
#include "ringbuffer.h"
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
  UDPCOMM UDP;                            // UDP cross daq communication server
  RINGBUFFER rbuffer;                          // ring buffer
  GPUCARD gcard;                          // GPU card
  FREQGEN fgen;                           // Freq generator
  LJACK ljack;                            // Labjack
  TWRITER twriter;			  //terminal writers

  if(argc>=2) {
    char fname_ini[256];
    init_settings(&settings,argv[1]);
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
  // GPU
  if (!settings.dont_process) gpuCardInit(&gcard,&settings);
  // UDP
  UDPCommInit(&UDP, &settings);
  // writer
  writerInit(&writer,&settings);
  // digitizer
  digiCardInit(&dcard,&settings);
  // ringBuffer 
  ringbufferInit(&rbuffer, &settings, &dcard);

  //MAIN LOOP
  digiWorkLoop(&dcard, &rbuffer, &gcard, &settings, &UDP,  &fgen, &ljack, &writer, &twriter);

  //shutdown
  digiCardCleanUp(&dcard, &settings);
  ringbufferCleanUp(&rbuffer);
  writerCleanUp(&writer);
  if (settings.fg_nfreq) freqGenCleanUp(&fgen);
  if (settings.lj_Non) LJCleanUp(&ljack);
  
  printf ("Done.\n");
  return 0;
}

