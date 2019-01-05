#pragma once
//Collects one sample's output and displays it all at once on terminal. 
//Returns cursor so that next cycle  will overwrite previous one on console.
#include "settings.h"
#define TERMINAL_LINES 25
#define TERMINAL_COLS 120
struct TWRITER{
    int terminal_nlines;
    int printEvery;
    int currentBlock;
    int debug;
  char lines[TERMINAL_LINES][TERMINAL_COLS];
};

void terminalWriterInit(TWRITER * t, SETTINGS *s);

//add formatted string to terminal buffer
void tprintfn (TWRITER * t, bool newline,  const char * fmt, ...);

//print to terminal and return cursor to beginning of output
void tflush(TWRITER * t); 

void terminalWriterCleanup(TWRITER * t); 
