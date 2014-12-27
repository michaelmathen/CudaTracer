C_FLAGS = -std=c99 
CXX_FLAGS = -g -std=c++11
NVCC_FLAGS =  -g 
LD_LIB_DIRS = -L/usr/local/cuda/lib64
LD_FLAGS = -lcuda -lcudart -lcurand 
LD_STATIC = 
INCLUDE_DIRS = -I/usr/local/cuda/include 

#CXX = /usr/local/bin/g++-4.8
#CC = /usr/local/bin/gcc-4.8
#NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
CC = gcc
NVCC = nvcc

OUTPUT_DIR=./bin/

CUDA_FILES = Renderer.cu \
             DistanceRenderer.cu \
             SceneAllocator.cu \
             PhongRenderer.cu

CUDA_OBJS = $(CUDA_FILES:%.cu=%.o)

CXX_FILES= main.cpp \
           ParseScene.cpp \
           ParsingException.cpp \
           SceneContainer.cpp

CXX_OBJS = $(CXX_FILES:%.cpp=%.o)

C_FILES= #bmpWrapper.c

C_OBJS = $(C_FILES:%.c=%.o)

PROGNAME=rayTracer

all: $(CXX_OBJS) $(C_OBJS) $(CUDA_OBJS)
	$(NVCC) $(NVCC_FLAGS) -dlink $(CUDA_OBJS) -o gpuCode.o
	$(CXX) $(CXX_FLAGS) gpuCode.o $(CUDA_OBJS) $(CXX_OBJS) $(C_OBJS) $(LD_LIB_DIRS) $(LD_FLAGS) -o $(PROGNAME) 

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -dc $<  -o $@

%.o:%.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_DIRS)  -c $<  -o $@

%.o:%.c 
	$(CC) $(C_FLAGS) -c $(INCLUDE_DIRS) -c $<  -o $@

clean: *.o
	rm *.o
