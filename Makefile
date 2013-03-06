
# Comment out the following line to generate release code
DEBUG = 1

# If you want to run cuda-gdb uncomment the -G option in the cuda debug options below
# Note that if you want to run on your own machine you may have to
# change the NVCCFLAGS to be an earlier streaming multiprocessor version
# Replacing 20 with 12 or 13 will most likely work
NVCCFLAGS   = -m64 #-arch=compute_20 
ifdef DEBUG
NVCCFLAGS   += -g -G
else
NVCCFLAGS   +=
endif

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
CUDA_HOME = /usr/local/cuda
CUDA_LIB = $(CUDA_HOME)/lib64
else
CUDA_HOME = /Developer/NVIDIA/CUDA-5.0
CUDA_LIB = $(CUDA_HOME)/lib
endif

GCCFLAGS    = #-arch x86_64
ifdef DEBUG
GCCFLAGS    += -g
else
GCCFLAGS    +=
endif

all: main.o ImageCleaner.o JPEGWriter.o CpuReference.o
	nvcc $(NVCCFLAGS) -L $(CUDA_LIB) -lcufft -lcudart -ljpeg -o ImageCleaner main.o ImageCleaner.o JPEGWriter.o CpuReference.o

main.o: main.cc
	g++ -I $(CUDA_HOME)/include -c -o main.o main.cc $(GCCFLAGS)

ImageCleaner.o: ImageCleaner.cu
	nvcc -c -o ImageCleaner.o ImageCleaner.cu $(NVCCFLAGS)

JPEGWriter.o: JPEGWriter.cc
	g++ -c -o JPEGWriter.o JPEGWriter.cc $(GCCFLAGS)

CpuReference.o: CpuReference.cc
	g++ -c -o CpuReference.o CpuReference.cc $(GCCFLAGS)

clean:
	rm -f *~ *.o *.linkinfo ImageCleaner
