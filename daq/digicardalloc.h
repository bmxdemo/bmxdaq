//memory allocation functions for allocating and freeing digitzer buffer
#pragma once

#include "spcm_examples/c_cpp/c_header/dlltyp.h"
#include <stdio.h>
#include <stdlib.h>


//allocate memory for digitizer buffer
void digiCardAlloc (int16* & data, int32 size);

//free memory from digitizer buffer
void digiCardFree (int16* & data);
