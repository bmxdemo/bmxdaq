/****

TERMINAL IO

Reverses cursor to allow continues overwrite over the terminal

******/


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "terminal.h"
#include "settings.h"

void terminalWriterInit(TWRITER * t, SETTINGS *s){
        t->terminal_nlines = 0;
	t->currentBlock=0;
	t->printEvery=s->print_every;
	if (t->printEvery==0) t->printEvery=1;
	t->debug=s->debug;
	for (int i=0; i<=TERMINAL_LINES; i++) {
	  printf("\n\033[K");
	  t->lines[i][0]='\0';
	}
}

void tprintfn(TWRITER * t, bool newline, const char* fmt, ...){
  if (!t) return; // if we pass nullptr, do nothing.
  if (t->terminal_nlines<TERMINAL_LINES) {
    char tmp[TERMINAL_COLS];
    va_list args;
    va_start(args,fmt);
    vsprintf(tmp,fmt,args);
    strcat(t->lines[t->terminal_nlines],tmp);
    if (newline) t->lines[++t->terminal_nlines][0]='\0';
    va_end(args);
  }
}


void tflush(TWRITER * t){ 
  if (t->currentBlock==0) {
      //while (t->terminal_nlines<t->num_lines) tprintfn(t,1,"");
      if (!t->debug)  printf("\033[%iA",TERMINAL_LINES);
      else printf ("--- *** --- \n");
      for (int i=0; i<TERMINAL_LINES; i++) {
	  printf("\033[K");
	  printf(t->lines[i]);
	  printf("\n");
      }
  }
  t->terminal_nlines = 0;
  t->lines[0][0]='\0';
  t->currentBlock++;
  if (t->currentBlock==t->printEvery)
    t->currentBlock=0;
} 
    
void terminalWriterCleanup(TWRITER * t)
{
  tflush(t);
  printf ("\n-----------------------------------\n");
}


