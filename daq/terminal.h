#pragma once
//Collects one sample's output and displays it all at once on terminal. 
//Returns cursor so that next cycle  will overwrite previous one on console.

struct TWRITER{
    int terminal_nlines;
    int num_lines;
    int printEvery;
    int currentBlock;
};

void terminalWriterInit(TWRITER * t, int num_lines, int printEvery);

//add formatted string to terminal buffer
void tprintfn (TWRITER * t, bool newline,  const char * fmt, ...);

//print to terminal and return cursor to beginning of output
void tflush(TWRITER * t); 

void terminalWriterCleanup(TWRITER * t); 
