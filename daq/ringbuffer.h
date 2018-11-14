#pragma once
#include "settings.h"
#include "terminal.h"
#include "writer.h"
#include "digicard.h"
#include <thread>

#define MAXCHUNKS 32
#define RBHEADERVERSION 1

struct RINGBUFFERHEADER {
  const char magic[8]=">>RBF<<"; // magic header char to recogize files *BMX*
  int version=RBHEADERVERSION;
  int ncards;
  size_t totbufsize;
};


struct RINGBUFFER {
  int8_t* buffer[2*MAXCHUNKS]; // pointer to numch
  int8_t* src[2]; // pointer to lastest src
  int num_chunks, cur_chunk[2];
  char fname_pattern[MAXFNLEN];
  char filename[MAXFNLEN];
  std::thread thread[2], dumpthread;
  bool filling[2];
  int fillremain;
  bool dumping;
  int dumpercent;
  size_t bufsize, ncards;
};

struct DIGICARD;

void ringbufferInit(RINGBUFFER *rb, SETTINGS *set, DIGICARD *dc);
void dumpRingBuffer(RINGBUFFER *rb);
void fillRingBuffer(RINGBUFFER *rb, int8_t* src[2]);
void printInfoRingBuffer(RINGBUFFER *rb, TWRITER* tw);
void ringbufferCleanUp(RINGBUFFER *rb);

