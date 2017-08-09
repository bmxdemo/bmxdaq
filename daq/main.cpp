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
  FREQGEN fgen;
  RFI rfi;                                //rfi stuff
  TWRITER ** twriter;			  //terminal writers

  if(argc>=2) {
    char fname_ini[256];
    sprintf(fname_ini,"%s",argv[1]);
    init_settings(&settings,fname_ini);
  } else
    init_settings(&settings,NULL);

  // intialize
  print_settings(&settings);
  digiCardInit(&dcard,&settings);
  twriter = (TWRITER **)malloc(settings.cuda_streams * sizeof(TWRITER *));
  for(int i =0; i < settings.cuda_streams ;  i++){
      twriter[i] = (TWRITER *)malloc(sizeof(TWRITER));
      terminalWriterInit(twriter[i], TERMINAL_BUFFER_SIZE, settings.print_every);
  }
  if (!settings.dont_process) gpuCardInit(&gcard,&settings);
  if (settings.fg_nfreq) freqGenInit(&fgen, &writer, &settings);
  writerInit(&writer,&settings);
  rfiInit(&rfi, &settings, &gcard);
  

  //work
  digiWorkLoop(&dcard, &gcard, &settings, &fgen, &writer, twriter,  &rfi);
  //shutdown
  digiCardCleanUp(&dcard, &settings);
  writerCleanUp(&writer);
  if (settings.fg_nfreq) freqGenCleanUp(&fgen);
  return 0;
}

