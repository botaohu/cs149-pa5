DEBUG ?= 1

SIZEX ?= 1024
SIZEY ?= 1024
CUFFT ?= 0

OBJECTS = main.o ImageCleaner.o JPEGWriter.o CpuReference.o

ifeq ($(shell uname), Linux)
CUDA_HOME = /usr/local/cuda
CUDA_LIB = $(CUDA_HOME)/lib64
CUDA_INC = $(CUDA_HOME)/include
else
CUDA_HOME = /Developer/NVIDIA/CUDA-5.0
CUDA_LIB = $(CUDA_HOME)/lib
CUDA_INC = $(CUDA_HOME)/include
endif

NVCCFLAGS = -arch=compute_20 -Xptxas "-v" -D SIZEX=$(SIZEX) -D SIZEY=$(SIZEY)
ifeq ($(DEBUG),1)
NVCCFLAGS += -g -G
else
NVCCFLAGS += 
endif

ifeq ($(shell uname), Darwin)
NVCCFLAGS += -m64 -code=sm_20
else
NVCCFLAGS += -code=sm_20
endif

NVCCLDFLAGS = -L $(CUDA_LIB)
NVCCLIBS = -lcudart -ljpeg


CFLAGS = -I $(CUDA_INC) -D SIZEX=$(SIZEX) -D SIZEY=$(SIZEY)
ifeq ($(DEBUG),1)
CFLAGS += -g
else
CFLAGS +=
endif

ifeq ($(shell uname), Darwin)
CFLAGS += -arch x86_64 
endif

ifeq ($(CUFFT), 1)
NVCCLIBS += -lcufft
CFLAGS += -D CUFFT=$(CUFFT)
NVCCFLAGS += -D CUFFT=$(CUFFT)
endif

.PHONY: all
all: $(OBJECTS)
	nvcc $(NVCCFLAGS) $(NVCCLDFLAGS) -o ImageCleaner $(OBJECTS) $(NVCCLIBS)

main.o: main.cc
	g++ $(CFLAGS) -c -o main.o main.cc

ImageCleaner.o: ImageCleaner.cu
	nvcc $(NVCCFLAGS) -c -o ImageCleaner.o ImageCleaner.cu

JPEGWriter.o: JPEGWriter.cc
	g++ $(CFLAGS) -c -o JPEGWriter.o JPEGWriter.cc

CpuReference.o: CpuReference.cc
	g++ $(CFLAGS) -c -o CpuReference.o CpuReference.cc

.PHONY: clean
clean:
	rm -f *~ *.o *.linkinfo ImageCleaner
