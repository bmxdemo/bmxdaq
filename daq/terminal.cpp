/****

TERMINAL IO

Reverses cursor to allow continues overwrite over the terminal

******/


#include <stdio.h>
#include <stdarg.h>
#include "terminal.h"
#include <stdlib.h>


void terminalWriterInit(TERMINALWRITER * t, int size){
	t->begin = (char *)malloc(size * sizeof(char));
	t->end = t->begin+size;
	t->current = t->begin;
	t->terminal_nlines = 0;
}

void terminalWriterAppend (TERMINALWRITER * t, bool newline, const char* fmt, ...){
    int remaining = t->end - t->current;
    if(remaining > 0){
	va_list args;
	va_start(args,fmt);
	t->current += vsnprintf(t->current, remaining, fmt, args);
	if(newline){
	    t->current += snprintf(t->current, remaining, "\n");
	    t->terminal_nlines++;
	}
	va_end(args);
    }
}
	
void terminalWriterPrint(TERMINALWRITER * t){
        t->current = t->begin;
        printf("%s",t->begin);
        printf("\033[%iA",t->terminal_nlines);
        t->terminal_nlines = 0;
} 
        
void terminalWriterCleanup(TERMINALWRITER * t){
	free(t->begin);
}
