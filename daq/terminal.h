#pragma once
//Collects one sample's output and displays it all at once on terminal. 
//Returns cursor so that next cycle  will overwrite previous one on console.
#include "settings.h"
#include <unistd.h>
#include <sys/select.h>
#include <termios.h>



#define TERMINAL_LINES 25
#define TERMINAL_COLS 120
struct TWRITER{
    int terminal_nlines;
    int printEvery;
    int currentBlock;
    int debug;
    char lines[TERMINAL_LINES][TERMINAL_COLS];
    struct termios orig_termios;

};

void terminalWriterInit(TWRITER * t, SETTINGS *s);

//add formatted string to terminal buffer
void tprintfn (TWRITER * t, bool newline,  const char * fmt, ...);

//print to terminal and return cursor to beginning of output
void tflush(TWRITER * t); 

// clean up and teturn terminal to normal;
void terminalWriterCleanup(TWRITER * t); 

void terminal_reset_mode(TWRITER *t);
void terminal_set_conio_mode(TWRITER *t);

// has a key been pressed?
int terminal_kbhit();
// if so, give it here
char terminal_getch();
