#
# Settings mak include things that you might need to make it work,
# eg gcc and NVCC settings, if you want to override
#
GPP=g++
NVCC=nvcc 
FFTWLINK=-lfftw3f
#SPCM DIR (to be set in settings.mak)
#SPCM_DIR= /path/to/spcm/
#SPCMLINK
SPCM_DIR=/opt/spcm
SPCM_LINK=-lspcm
-include settings.mak
SPCM_INCLUDE=-I$(SPCM_DIR)
SPCM_OBJ=$(SPCM_DIR)/spcm_examples/c_cpp//common/ostools/spcm_ostools_linux.o


HEADERFILES=settings.h digicard.h  gpucard.h writer.h
SOURCES=main.cpp settings.cpp digicard.cpp  writer.cpp
CUDASOURCES=gpucard.cu
OBJS=$(SOURCES:.cpp=.o) $(CUDASOURCES:.cu=.o) 


all: daq.out

%.o: %.cpp $(HEADER_FILES)
	$(GPP) -Ofast -Wno-format-security $(SPCM_INCLUDE) -c -o $@ $<

%.o: %.cu $(HEADER_FILES)
	$(NVCC)  -O3 -c $(SPCM_INCLUDE) -o $@ $<

daq.out: $(OBJS)
	$(NVCC) -o $@ $(OBJS) -L. -lm -lcufft $(SPCM_OBJ) $(SPCM_LINK)

clean:
	rm *.o
