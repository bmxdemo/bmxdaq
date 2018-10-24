/****

TERMINAL IO

Reverses cursor to allow continues overwrite over the terminal

******/


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "terminal.h"


void terminalWriterInit(TWRITER * t, int printEvery){
        t->terminal_nlines = 0;
	t->currentBlock=0;
	if (printEvery==0) printEvery=1;
	t->printEvery = printEvery;
	printf ("-----------------------------------\n");
}

void tprintfn(TWRITER * t, bool newline, const char* fmt, ...){
  if (t->currentBlock==0) {
    va_list args;
    va_start(args,fmt);
    vprintf(fmt,args);
    if (newline) {
      t->terminal_nlines++;
      printf ("\n");
    }
    va_end(args);
  }
}


void tflush(TWRITER * t){ 
  if (t->currentBlock==0) {
    printf("\033[%iA",t->terminal_nlines);
    t->terminal_nlines = 0;
  }
  t->currentBlock++;
  if (t->currentBlock==t->printEvery)
    t->currentBlock=0;
} 
    
void terminalWriterCleanup(TWRITER * t)
{
  printf ("\n\n\n-----------------------------------\n");
}


