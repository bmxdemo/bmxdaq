/****

TERMINAL IO

Reverses cursor to allow continues overwrite over the terminal

******/


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "terminal.h"
#include "settings.h"

void terminalWriterInit(TWRITER * t, SETTINGS *s){
        t->terminal_nlines = 0;
	t->currentBlock=0;
	t->num_lines=25;
	t->printEvery=s->print_every;
	if (t->printEvery==0) t->printEvery=1;
	t->debug=s->debug;
	for (int i=0; i<=t->num_lines; i++) printf("\n\033[K");
	printf("\033[%iA",t->num_lines);

}

void tprintfn(TWRITER * t, bool newline, const char* fmt, ...){
  if ((t->currentBlock==0)&&(t->terminal_nlines<t->num_lines)) {
    va_list args;
    va_start(args,fmt);
    vprintf(fmt,args);
    if (newline) {
      t->terminal_nlines++;
      printf ("\n");
      printf("\033[K");

    }
    va_end(args);
  }
}


void tflush(TWRITER * t){ 
  if (t->currentBlock==0) {
    if (!t->debug) {
      while (t->terminal_nlines<t->num_lines) tprintfn(t,1,"");
      printf("\033[%iA",t->num_lines);
    } else printf ("--- *** --- \n");
      t->terminal_nlines = 0;
  }
  t->currentBlock++;
  if (t->currentBlock==t->printEvery)
    t->currentBlock=0;
} 
    
void terminalWriterCleanup(TWRITER * t)
{
  tflush(t);
  printf ("\n\n\n-----------------------------------\n");
}


