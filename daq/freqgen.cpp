#include "freqgen.h"
#include "terminal.h"
#include "rs232.h"
#include <stdlib.h>

void freqGenInit (FREQGEN *fg, WRITER* wr, SETTINGS *set) {
  printf ("\n\nInitializing tone generator\n");
  printf ("===============================\n");
  fg->cport=RS232_GetPortnr(set->fg_port);
  if (fg->cport<0) {
    printf ("Port %s not found. \n",set->fg_port);
    exit(1);
  }
  char mode[]={'8','N','1',0};
  if(RS232_OpenComport(fg->cport, set->fg_baudrate, mode)) {
    printf("Can not open comport\n");
    exit(1);
  }
  RS232_cputs(fg->cport, "*IDN?\n");
  usleep(100000);  
  unsigned char buf[4096];
  int n = RS232_PollComport(fg->cport, buf, 4095);
  buf[n] = 0;
  printf ("Found device: %s\n",buf);

  fg->ccount=0; fg->cswitch=set->fg_switchevery;
  fg->cfreq=0; fg->nfreq=set->fg_nfreq;
  for (int i=0; i<fg->nfreq; i++) {
    fg->freq[i]=set->fg_freq[i];
    fg->ampl[i]=set->fg_ampl[i];
  }
  freqGenLoop(fg,wr);
  treturn();
}


//main worker loop
void freqGenLoop (FREQGEN *fg, WRITER* wr) {
  tprintfn("ToneGen: count %i/%i freq #%i %fMHz %fVpp",fg->ccount, 
	   fg->cswitch, fg->cfreq, fg->freq[fg->cfreq], fg->ampl[fg->cfreq]);
  if (fg->ccount==0) {
    char buf[128];
    sprintf(buf,"FREQ%fMHz\n",fg->freq[fg->cfreq]);
    RS232_cputs(fg->cport, buf);
    sprintf(buf,"AMPR%fVPp\n",fg->ampl[fg->cfreq]);
    RS232_cputs(fg->cport, buf);
    if (wr)
      wr->tone_freq=fg->freq[fg->cfreq];
    fg->cfreq = (++fg->cfreq)%fg->nfreq;
  }
  fg->ccount = (++fg->ccount)%fg->cswitch;
}

//shutdown
void freqGenCleanUp(FREQGEN *fg) {
  RS232_CloseComport(fg->cport);
}
