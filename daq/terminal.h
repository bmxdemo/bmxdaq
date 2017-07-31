#pragma once
//Collects one sample's output and displays it all at once on terminal. 
//Returns cursor so that next cycle  will overwrite previous one on console.

struct TERMINALWRITER{
    char * begin;
    char * end;
    char * current;
    int terminal_nlines;
};

void terminalWriterInit(TERMINALWRITER * t, int size);
//add formatted string to terminal buffer
void terminalWriterAppend (TERMINALWRITER * t, bool newline,  const char * fmt, ...);
//print to terminal and return cursor to beginning of output
void terminalWriterPrint(TERMINALWRITER * t);
void terminalWriterCleanup(TERMINALWRITER * t);


// Like printf, but automatically adds a new line and increment line count
 void tprintfn(const char* fmt, ...);
// // returen up line count lines
 void treturn();
