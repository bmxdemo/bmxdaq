/****

TERMINAL IO

Reverses cursor to allow continues overwrite over the terminal

******/


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "terminal.h"


void terminalWriterInit(TWRITER * t, int size){
        t->begin = (char *)malloc(size * sizeof(char));
        t->end = t->begin+size;
        t->current = t->begin;
        t->terminal_nlines = 0;
}

void tprintfn(TWRITER * t, bool newline, const char* fmt, ...){
    int remaining = t->end - t->current;
    if(remaining > 0){ 
        va_list args;
        va_start(args,fmt);
        t->current += vsnprintf(t->current, remaining, fmt, args);
        if(newline){
	    remaining = t->end - t->current;
            if(remaining >1){
	    	t->current += snprintf(t->current, remaining, "\n");
	    	t->terminal_nlines++;
	    }
        }   
        va_end(args);
    }   
}    

void tflush(TWRITER * t){ 
        t->current = t->begin;
        printf("%s",t->begin);
        printf("\033[%iA",t->terminal_nlines);
        t->terminal_nlines = 0;
} 
    
void terminalWriterCleanup(TWRITER * t){ 
        free(t->begin);
}


