#pragma once


/*
**************************************************************************
bDoCardSetuo: setup matching the calculation routine
**************************************************************************
*/

#include "settings.h"
#include "writer.h"

struct LJACK {
  int num_on,num_off,num_tot;
  int counter;
  int handle;
  double voltage0;
  int diode;
};


//initialize
void LJInit (LJACK *lj, WRITER* wr, SETTINGS *set);

//main worker loop
void LJLoop (LJACK *lj, WRITER* wr);

//shutdown
void LJCleanUp(LJACK *lj);
