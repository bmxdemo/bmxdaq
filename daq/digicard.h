#pragma once

#include "settings.h"
#include "UDPCommunication.h"
#include "ringbuffer.h"
#include "gpucard.h"
#include "freqgen.h"
#include "ljack.h"
#include "writer.h"
#include "terminal.h"
#include "spcm_examples/c_cpp/c_header/dlltyp.h"
#include "spcm_examples/c_cpp/c_header/regs.h"
#include "spcm_examples/c_cpp/c_header/spcerr.h"
#include "spcm_examples/c_cpp/c_header/spcm_drv.h"

// ----- standard c include files -----
#include <stdio.h>
#include <string.h>
#include <stdlib.h>



/*
**************************************************************************
bDoCardSetuo: setup matching the calculation routine
**************************************************************************
*/

struct DIGICARD {
  drv_handle  hCard[2];
  int32       serialNumber[2];
  int16**     pnData;
  int         two_channel;
  int         num_cards;
  int32       lNotifySize;
  int32       lBufferSize;
};


//initialize
void digiCardInit (DIGICARD *card, SETTINGS *set);

//main worker loop
void  digiWorkLoop(DIGICARD *card, RINGBUFFER *rb, GPUCARD *gcard, SETTINGS *set, UDPCOMM *UDP,
		   FREQGEN *fgen, LJACK *lj, WRITER *w, TWRITER *t);

//shutdown
void digiCardCleanUp(DIGICARD *card, SETTINGS *set);
