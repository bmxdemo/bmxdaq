#include "settings.h"
#include "digicard.h"
#include "digicardalloc.h"
#include "gpucard.h"
#include "terminal.h"
#include "freqgen.h"

// ----- include standard driver header from library -----
#include "spcm_examples/c_cpp/common/ostools/spcm_oswrap.h"
#include "spcm_examples/c_cpp/common/ostools/spcm_ostools.h"


// ----- standard c include files -----
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>

#include <thread>
/* Ctrl+C hander */

volatile sig_atomic_t stopSignal = 0;

void loop_signal_handler(int sig){ // can be called asynchronously
  stopSignal=1;
}


/*
**************************************************************************
szTypeToName: doing name translation
**************************************************************************
*/


char* szTypeToName (int32 lCardType){
  static char szName[50];
  switch (lCardType & TYP_SERIESMASK){
    case TYP_M2ISERIES:     sprintf (szName, "M2i.%04x", (unsigned int) (lCardType & TYP_VERSIONMASK));      break;
    case TYP_M2IEXPSERIES:  sprintf (szName, "M2i.%04x-Exp", (unsigned int) (lCardType & TYP_VERSIONMASK));  break;
    case TYP_M3ISERIES:     sprintf (szName, "M3i.%04x", (unsigned int) (lCardType & TYP_VERSIONMASK));      break;
    case TYP_M3IEXPSERIES:  sprintf (szName, "M3i.%04x-Exp", (unsigned int) (lCardType & TYP_VERSIONMASK));  break;
    case TYP_M4IEXPSERIES:  sprintf (szName, "M4i.%04x-x8", (unsigned int) (lCardType & TYP_VERSIONMASK));   break;
    case TYP_M4XEXPSERIES:  sprintf (szName, "M4x.%04x-x4", (unsigned int) (lCardType & TYP_VERSIONMASK));   break;
    default:                sprintf (szName, "unknown type");                               break;
  }
  return szName;
}



void printErrorDie(const char* message, DIGICARD *card, int cardIndex,  SETTINGS *set) {
  char szErrorTextBuffer[ERRORTEXTLEN];
  spcm_dwGetErrorInfo_i32 (card->hCard[cardIndex], NULL, NULL, szErrorTextBuffer);
  //correct card number in case where we are only using the second digitizer card
  int cardNum = (set->card_mask==2)? 1 : cardIndex;
  printf ("Digitizer card %d fatal error: %s\n",cardNum, message);
  printf ("Error Text: %s\n", szErrorTextBuffer);
  digiCardCleanUp(card, set);
  exit(1);
}

void startDAQ(uint32 & dwError, drv_handle & hCard){
  dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_START |
    M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA);
}


/*
**************************************************************************
Setup the digitizer card
**************************************************************************
*/

void digiCardInit (DIGICARD *card, SETTINGS *set) {
  int numCards = 1 + (set->card_mask==3); //how many digitizer cards we are using
  printf ("Allocating digitizer buffer...\n");
  /// now set the memory
  card->two_channel = (set->channel_mask==3);
  card->lNotifySize = set->fft_size*(1+card->two_channel);
  card->lBufferSize = card->lNotifySize*set->buf_mult;
    
  /// alocate buffer
  card->pnData=(int16**)malloc(2*sizeof(int16*));
  for(int i=0; i<numCards; i++){
    digiCardAlloc(card->pnData[i], card->lBufferSize);
    if (!card->pnData[i]){
      printf ("memory allocation failed\n");
      exit(1);
    }
  }

  if(set->simulate_digitizer){
    printf ("**Not using real card, simulating...**\n");
    // Filling buffer
    printf ("Filling Fake buffer...\n");
    for(int n=0; n<numCards; n++){
      int8_t ch1lu[64], ch2lu[64];
      for(int i=0; i<64; i++) {
        ch1lu[i]=-1;//int(20*cos(2*2*M_PI*i/64)+10*sin(2*M_PI*i/64));
        ch2lu[i]=-1;//31+i;
      }
      int i=0;
      int32_t s=card->lBufferSize;
      int8_t *data=(int8_t*) card->pnData[n];
      printf ("buffer size=%i\n",s);
      for (int32_t k=0; k<s-1; k+=2) {
        data[k]=ch1lu[i];
        data[k+1]=ch2lu[i];
        i+=1;
        if (i==64) i=0;
      }
    }
    printf ("Simulated digitizer card and buffer ready.\n");
    return;
  }
  
  printf ("\n\nInitializing digitizer\n");
  printf ("==========================\n");

  //open digitizer cards
  switch(set->card_mask){
    case 1:
      card->hCard[0] = spcm_hOpen ((char *)"/dev/spcm0");
      break;
    case 2:
      card->hCard[0] = spcm_hOpen ((char *)"/dev/spcm1");
      break;
    case 3:
      card->hCard[0] = spcm_hOpen ((char *)"/dev/spcm0");
      card->hCard[1] = spcm_hOpen ((char *)"/dev/spcm1");
      break;
    default:
      printf("invalid value for card_mask.\n");
      exit(1);
  }
  
  for(int i =0; i < numCards; i++){
    if (!card->hCard[i]) printErrorDie("Can't open digitizer card", card, i, set);
    int32 lCardType, lSerialNumber, lFncType;
    // read type, function and sn and check for A/D card
    spcm_dwGetParam_i32 (card->hCard[i], SPC_PCITYP,         &lCardType);
    spcm_dwGetParam_i32 (card->hCard[i], SPC_PCISERIALNO,    &lSerialNumber);
    spcm_dwGetParam_i32 (card->hCard[i], SPC_FNCTYPE,        &lFncType);
    switch (lFncType){
      case SPCM_TYPE_AI:
        printf ("Found: %s sn %05d\n", szTypeToName (lCardType), lSerialNumber);
        break;

      default:
        printf ("Card: %s sn %05d not supported. \n", szTypeToName (lCardType), lSerialNumber);            
        exit(1);
    }

    // do a simple standard setup
    // always do two channels
    spcm_dwSetParam_i32 (card->hCard[i], SPC_CHENABLE,       set->channel_mask);     // just 1 channel enabled
    spcm_dwSetParam_i32 (card->hCard[i], SPC_PRETRIGGER,     1024);                  // 1k of pretrigger data at start of FIFO mode
    spcm_dwSetParam_i32 (card->hCard[i], SPC_CARDMODE,       SPC_REC_FIFO_SINGLE);   // single FIFO mode
    spcm_dwSetParam_i32 (card->hCard[i], SPC_TIMEOUT,        5000);                  // timeout 5 s
    spcm_dwSetParam_i32 (card->hCard[i], SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);    // trigger set to software
    spcm_dwSetParam_i32 (card->hCard[i], SPC_TRIG_ANDMASK,   0);                     // ...
    if (set->ext_clock_mode) 
      spcm_dwSetParam_i32 (card->hCard[i], SPC_CLOCKMODE,      SPC_CM_EXTREFCLOCK);
    else
      spcm_dwSetParam_i32 (card->hCard[i], SPC_CLOCKMODE,      SPC_CM_INTPLL);       // clock mode internal PLL

    spcm_dwSetParam_i64 (card->hCard[i], SPC_REFERENCECLOCK, set->spc_sample_rate);
    spcm_dwSetParam_i64 (card->hCard[i], SPC_SAMPLERATE, set->spc_ref_clock);

    spcm_dwSetParam_i32 (card->hCard[i], SPC_CLOCKOUT,       0);                     // no clock output

    spcm_dwSetParam_i32 (card->hCard[i], SPC_AMP0,  set->ADC_range  );
    spcm_dwSetParam_i32 (card->hCard[i], SPC_AMP1,  set->ADC_range  );
    int32_t range1, range2;
    spcm_dwGetParam_i32 (card->hCard[i], SPC_AMP0,  &range1 );
    spcm_dwGetParam_i32 (card->hCard[i], SPC_AMP1,  &range2 );
    printf ("ADC ranges for CH1/2: %i/%i mV\n",range1,range2);
    long long int srate;
    spcm_dwGetParam_i64 (card->hCard[i], SPC_SAMPLERATE, &srate);
    printf ("Sampling rate set to %.1lf MHz\n", srate/1000000.); 
   
    // define transfer
    spcm_dwDefTransfer_i64 (card->hCard[i], SPCM_BUF_DATA, SPCM_DIR_CARDTOPC,
        card->lNotifySize, card->pnData[i], 0, card->lBufferSize);
  } 
  printf ("Digitizer card and buffer ready.\n");
}

float deltaT (timespec t1,timespec t2) {
  return ( t2.tv_sec - t1.tv_sec )
    + ( t2.tv_nsec - t1.tv_nsec )/ 1e9;
}

void  digiWorkLoop(DIGICARD *dc, GPUCARD *gc, SETTINGS *set, FREQGEN *fgen, LJACK *lj,
       WRITER *w, RFI *rfi) {
  printf ("\n\nStarting main loop\n");
  printf ("==========================\n");
  
  int numCards = 1 + (set->card_mask==3); //how many digitizer cards we are using
  
  uint32      dwError[2];
  int32       lStatus[2], lAvailUser[2], lPCPos[2], fill[2];
  int8_t * bufstart[2];
  
  // start everything
  if(!set->simulate_digitizer){
    std::thread th[2];
    for(int i=1; i<numCards; i++)
      th[i] = std::thread(startDAQ, std::ref(dwError[i]), std::ref(dc->hCard[i]));
    for(int i=1; i<numCards; i++)
      th[i].join();
      
    // check for error
    for(int i=0; i<numCards; i++)
      if (dwError[i] != ERR_OK) printErrorDie("Cannot start FIFO\n",dc, i, set);
  }
  else
    dwError[0]=dwError[1] = ERR_OK;

  
  struct timespec timeStart, timeNow, tSim, t1;
  int sim_ofs=0;
  clock_gettime(CLOCK_REALTIME, &timeStart);
  tSim=timeStart;
  fill[0]=fill[1]=69;
  float towait=set->fft_size/set->sample_rate;
  long int sample_count=0;
  signal(SIGINT, loop_signal_handler);
  while (!stopSignal) {
    clock_gettime(CLOCK_REALTIME, &t1);
    float dt=deltaT(tSim,t1);
    tprintfn ("Cycle taking %fs, hope for < %fs",dt, towait);
    if (set->simulate_digitizer) {
      for(int i=0; i<numCards; i++){
        lPCPos[i] = dc->lNotifySize*sim_ofs;
        sim_ofs = (sim_ofs+1)%set->buf_mult;
        lAvailUser[i]=dc->lNotifySize;
        if (deltaT(tSim,t1)>towait) {
          fill[i]+=30;
        }
        else
          do {
          clock_gettime(CLOCK_REALTIME, &t1);
          if (fill[i]>69) fill[i]-=30;
          } while (deltaT(tSim,t1)<towait);
      }
    }
    else {
      t1=tSim;
      for (int i = 0; i < numCards; i++){
        dwError[i] = spcm_dwSetParam_i32 (dc->hCard[i], SPC_M2CMD, M2CMD_DATA_WAITDMA);
        if (dwError[i] != ERR_OK) printErrorDie ("DMA wait fail\n",dc, i, set);
        spcm_dwGetParam_i32 (dc->hCard[i], SPC_M2STATUS,             &lStatus[i]);
        spcm_dwGetParam_i32 (dc->hCard[i], SPC_DATA_AVAIL_USER_LEN,  &lAvailUser[i]);
        spcm_dwGetParam_i32 (dc->hCard[i], SPC_DATA_AVAIL_USER_POS,  &lPCPos[i]);
        spcm_dwGetParam_i32 (dc->hCard[i], SPC_FILLSIZEPROMILLE,  &fill[i]);
        clock_gettime(CLOCK_REALTIME, &tSim);
        dt=deltaT(t1,tSim);
        tprintfn ("Measured dt for card %d: %f ms, rate=%f MHz",i, dt*1e3, set->fft_size/dt/1e6);
      }
    }
    if (lAvailUser[0] >= dc->lNotifySize && (numCards ==1 || lAvailUser[1] >= dc->lNotifySize)){
        clock_gettime(CLOCK_REALTIME, &timeNow);
        double accum = deltaT(timeStart, timeNow);
        for(int i=0; i<numCards; i++){
            tprintfn("Time: %fs; Status:%i; Pos:%08x; digitizer buffer fill %i/1000   ", 
                accum, lStatus[i], lPCPos[i],fill[i]);
            bufstart[i]=((int8_t*)dc->pnData[i]+lPCPos[i]);
        }
        if (set->dont_process) 
          tprintfn (" ** no GPU processing");
        else{
            gpuProcessBuffer(gc,bufstart,w,rfi, set);
        }

        // tell driver we're done
        if (!set->simulate_digitizer)
          for(int i = 0; i < numCards; i++)
            spcm_dwSetParam_i32 (dc->hCard[i], SPC_DATA_AVAIL_CARD_LEN, dc->lNotifySize);
        
        // drive frequency generator if needed
        if (set->fg_nfreq) freqGenLoop(fgen, w);
        // drive labjack
        if (set->lj_Non) LJLoop(lj,w);
        // write waveform if requested
        if (set->wave_nbytes>0) {
          tprintfn ("filename=%s",set->wave_fname);
          FILE *fw=fopen(set->wave_fname,"wb");
          if (fw!=NULL) {
            for(int i=0; i< numCards; i++)
              fwrite(bufstart[i],sizeof(int8_t),set->wave_nbytes,fw);
            fclose(fw);
          }
        }
        // break if sufficient number of samples
        if ((++sample_count) == set->nsamples) break;

        // return terminal cursor
        treturn();
    }
  }   
  
  printf("\n\n\n\n\n\n\n\n\n");
  if (stopSignal) printf ("Ctrl-C detected. Stopping.\n");
  if (sample_count==set->nsamples) printf ("Reached required number of samples.\n");
  //Write last digitizer bufer to a file 
  if(set->print_last_buffer){
    printf("Printing last digitizer buffer to a file...\n");
    for(int i=0; i < numCards; i++)
      writerWriteLastBuffer(w, bufstart, numCards, dc->lNotifySize);
  }
  printf ("Stoping digitizer FIFO...\n");
  // send the stop command
  for(int i=0; i < numCards; i++){
    dwError[i] = set->simulate_digitizer ? ERR_OK :
        spcm_dwSetParam_i32 (dc->hCard[i], SPC_M2CMD, M2CMD_CARD_STOP | 
       M2CMD_DATA_STOPDMA);
    if (dwError[i] != ERR_OK) printErrorDie("Error stopping card.\n",dc, i, set);
  }
}


void digiCardCleanUp(DIGICARD *card, SETTINGS *set) {
  printf ("Closing digitizer... \n");
  int numCards = 1 + (set->card_mask==3); //how many digitizer cards we are using
  for(int i =0; i < numCards; i++)
    digiCardFree(card->pnData[i]);
  if (!set->simulate_digitizer)
    for(int i =0; i < numCards; i++)
      spcm_vClose (card->hCard[i]);
}
