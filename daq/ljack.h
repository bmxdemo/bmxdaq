#pragma once


/*
**************************************************************************
bDoCardSetuo: setup matching the calculation routine
**************************************************************************
*/

#include "settings.h"
#include "writer.h"
#include "terminal.h"

struct LJACK {
  int num_on,num_off,num_tot;
  int counter;
  int handle;
  double voltage0;
  int diode;
};


//initialize
void LJInit (LJACK *lj, WRITER* wr,SETTINGS *set);

//main worker loop
void LJLoop (LJACK *lj, WRITER* wr, TWRITER * twr);

//shutdown
void LJCleanUp(LJACK *lj);
