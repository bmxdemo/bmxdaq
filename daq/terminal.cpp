/****

TERMINAL IO

Reverses cursor to allow continues overwrite over the terminal

******/


#include <stdio.h>
#include <stdarg.h>
#include "terminal.h"

int terminal_nlines=0;

void tprintfn(const char* fmt, ...)
{
    va_list args;
    va_start(args,fmt);
    vprintf(fmt,args);
    printf("\n");
    terminal_nlines++;
    va_end(args);
}

void treturn() {
  printf("\033[%iA",terminal_nlines);
  terminal_nlines=0;
}

