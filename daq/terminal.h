#pragma once
//Collects one sample's output and displays it all at once on terminal. 
//Returns cursor so that next cycle  will overwrite previous one on console.

struct TWRITER{
    char * begin;
    char * end;
    char * current;
    int terminal_nlines;
};

void terminalWriterInit(TWRITER * t, int size);

//add formatted string to terminal buffer
void tprintfn (TWRITER * t, bool newline,  const char * fmt, ...);

//print to terminal and return cursor to beginning of output
void tflush(TWRITER * t); 

void terminalWriterCleanup(TWRITER * t); 
