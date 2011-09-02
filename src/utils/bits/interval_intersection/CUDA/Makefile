CPP=g++
CC=gcc
CFLAGS=-g
HOME=/home/rl6sf
NVCC=/usr/local/cuda/bin/nvcc
SDK_PATH=$(HOME)/NVIDIA_GPU_Computing_SDK
SDK_COMMON_INC=$(SDK_PATH)/C/common/inc
SDK_SHARED_INC=$(SDK_PATH)/C/shared/inc
CUDA_INCLUDE=/usr/local/cuda/include
TIMER_PATH=$(HOME)/src/utils/c/timer
RADIX_SORT_PATH=$(HOME)/src/utils/cuda/radixsort
REDUCE_PATH=$(HOME)/src/utils/cuda/reduce
MT_PATH=$(HOME)/src/utils/c/mt
BSEARCH_PATH=$(HOME)/src/binary_search/lib
INTERVAL_INTERSECTION_PATH=$(HOME)/src/interval_intersection/lib

LFLAGS_A=-fPIC
LFLAGS_B=-L/usr/local/cuda/lib64 \
		-L$(SDK_PATH)/C/lib \
		-L$(SDK_PATH)/C/common/lib/linux \
		-L$(SDK_PATH)/shared/lib \
		-lcudart \
		-lcudpp_x86_64 


INCLUDES=-I$(INTERVAL_INTERSECTION_PATH) \
		 -I$(BSEARCH_PATH)\
		 -I$(MT_PATH)\
		 -I$(TIMER_PATH)\
		 -I$(RADIX_SORT_PATH) \
		 -I$(REDUCE_PATH) \
		 -I$(CUDA_INCLUDE) \
		 -I$(SDK_SHARED_INC) \
		 -I$(SDK_COMMON_INC) 

LIB_OBJS=$(INTERVAL_INTERSECTION_PATH)/interval.o \
		 $(INTERVAL_INTERSECTION_PATH)/interval_cuda.cu.o \
		 $(BSEARCH_PATH)/bsearch.o \
		 $(MT_PATH)/mt.o \
		 $(TIMER_PATH)/timer.o \
		 $(REDUCE_PATH)/reduce.cu.o \
		 $(RADIX_SORT_PATH)/radixsort.cu.o \
		 $(RADIX_SORT_PATH)/radixsort.o \

C_SRC=$(INTERVAL_INTERSECTION_PATH)/interval.c \
		$(BSEARCH_PATH)/bsearch.c \
		$(MT_PATH)/mt.c \
		$(TIMER_PATH)/timer.c 

.SUFFIXES: .c .cpp .cu .o

EXECS=count_interval_intersections_cuda \
	  per_interval_count_interval_intersections_cuda

all: $(EXECS)

#count_interval_intersections_cuda: count_interval_intersections_cuda.cu.o \
		#$(LIB_OBJS)
	#$(CPP) $(LFLAGS_A) -o $@ \
		#$(INCLUDES) \
		#$(C_SRC) \
   		#$(LIB_OBJS) \
		#count_interval_intersections_cuda.cu.o \
		#$(LFLAGS_B) 

%: %.cu.o $(LIB_OBJS)
	$(CPP) $(LFLAGS_A) -o $@ \
		$(C_SRC) $(INCLUDES) $+ $(LFLAGS_B)


%.cu.o: %.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	
%.o: %.cpp
	$(CPP) $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(EXECS) *.a *.o $(LIB_OBJS)
