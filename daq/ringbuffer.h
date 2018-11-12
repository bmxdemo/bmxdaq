#pragma once
#include "settings.h"
#include "terminal.h"
#include "writer.h"
#include "digicard.h"
#include <thread>

#define MAXCHUNKS 32

struct RINGBUFFER {
  int8_t* buffer[2*MAXCHUNKS]; // pointer to numch
  int num_chunks, cur_chunk[2];
  char fname_pattern[MAXFNLEN];
  std::thread thread[2];
  bool filling[2];
  bool dumping;
  size_t bufsize, ncards;
};

struct DIGICARD;

void ringbufferInit(RINGBUFFER *rb, SETTINGS *set, DIGICARD *dc);
void dumpRingBuffer(RINGBUFFER *rb);
void fillRingBuffer(RINGBUFFER *rb, int cardnum, int8_t* src);
void printInfoRingBuffer(RINGBUFFER *rb, TWRITER* tw);
void ringbufferCleanUp(RINGBUFFER *rb);

