#include "ljack.h"
#include "terminal.h"
#include <stdlib.h>
#include <LabJackM.h> 

void ErrorCheck(int err, const char * formattedDescription, ...);

void LJInit (LJACK *lj, WRITER* wr, SETTINGS *set) {
  printf ("\n\nInitializing LabJack Interface\n");
  printf ("===============================\n");
  lj->num_off = set->lj_Noff;
  lj->num_on = set->lj_Non;
  lj->num_tot = lj->num_on + lj->num_off;
  lj->counter=0;
  
  // Open first found LabJack                                                                                                                                              
  int err = LJM_Open(LJM_dtANY, LJM_ctANY, "LJM_idANY", &lj->handle);
  ErrorCheck(err, "LJM_Open");  

  double value=10.0;  // +- 10 volts range                                                                                                                                        
  err = LJM_eWriteAddress(lj->handle, 40000, LJM_FLOAT32, 10.0);                                                                                                                     
  ErrorCheck(err, "LJM Set voltage");    
  LJLoop(lj,wr, NULL);
}


//main worker loop
void LJLoop (LJACK *lj, WRITER* wr, TWRITER* twr) {
  int err=0;
  if (lj->counter==0) {
    //switch off
    err = LJM_eWriteAddress(lj->handle, 1000, LJM_FLOAT32, 0.0);  //DAC
    ErrorCheck(err, "LJM_eWriteAddress");
    lj->diode=0;
  } else if (lj->counter==lj->num_off) {
    // switch on
    err = LJM_eWriteAddress(lj->handle, 1000, LJM_FLOAT32, 5.0);  //DAC
    ErrorCheck(err, "LJM_eWriteAddress");
    lj->diode=1;
  }
  // read voltage3
  err = LJM_eReadAddress(lj->handle, 0, LJM_FLOAT32, &lj->voltage0);  // yes 0 is AIN0  

  if (wr) {
    wr->lj_voltage0=lj->voltage0;
    //wr->lj_diode=lj->diode;
  }
  if (twr)
    tprintfn (twr, 1, "LabJack: count %i/%i Diode:%i  V0:%g ",lj->counter, 
	   lj->num_tot, lj->diode, lj->voltage0);

  lj->counter = (++lj->counter)%(lj->num_tot);

}

//shutdown
void LJCleanUp(LJACK *lj) {
  int err = LJM_Close(lj->handle);
  ErrorCheck(err, "LJM_Close");
}



// from LJ examples

#define INITIAL_ERR_ADDRESS -2
// This is just something negative so normal addresses are not confused with it

#include <stdarg.h>

typedef enum {
  ACTION_PRINT_AND_EXIT,
        ACTION_PRINT
} ErrorAction;

void PrintErrorAddressHelper(int errAddress)
{
   if (!(errAddress < 0))
     printf("\terror address: %d\n", errAddress);
}


// The "internal" print function for ErrorCheck and ErrorCheckWithAddress
void _ErrorCheckWithAddress(int err, int errAddress, ErrorAction action,
			    const char * description, va_list args)
{
  char errName[LJM_MAX_NAME_SIZE];
  if (err >= LJME_WARNINGS_BEGIN && err <= LJME_WARNINGS_END) {
    LJM_ErrorToString(err, errName);
    vfprintf (stdout, description, args);
    printf(" warning: \"%s\" (Warning code: %d)\n", errName, err);
    PrintErrorAddressHelper(errAddress);
  }
  else if (err != LJME_NOERROR)
    {
      LJM_ErrorToString(err, errName);
      vfprintf (stdout, description, args);
      printf(" error: \"%s\" (ErrorCode: %d)\n", errName, err);
      PrintErrorAddressHelper(errAddress);

      if (action == ACTION_PRINT_AND_EXIT) {
	printf("Closing all devices and exiting now\n");
	LJM_CloseAll();
	exit(err);
      }
    }
}


void ErrorCheck(int err, const char * formattedDescription, ...)
{
  va_list args;

  va_start (args, formattedDescription);
  _ErrorCheckWithAddress(err, INITIAL_ERR_ADDRESS, ACTION_PRINT_AND_EXIT,
			 formattedDescription, args);
  va_end (args);
}
