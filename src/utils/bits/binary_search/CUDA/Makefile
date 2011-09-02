CPP=g++
CC=gcc
HOME=/home/rl6sf
NVCC=/usr/local/cuda/bin/nvcc
SDK_PATH=$(HOME)/NVIDIA_GPU_Computing_SDK
SDK_COMMON_INC=$(SDK_PATH)/C/common/inc
SDK_SHARED_INC=$(SDK_PATH)/C/shared/inc
CUDA_INCLUDE=/usr/local/cuda/include
TIMER_PATH=$(HOME)/src/utils/c/timer
RADIX_SORT_PATH=$(HOME)/src/utils/cuda/radixsort
REDUCE_PATH=$(HOME)/src/utils/cuda/reduce
RANDOM_PATH=$(HOME)/src/utils/cuda/random
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


INCLUDES=-I$(BSEARCH_PATH)\
		 -I$(MT_PATH)\
		 -I$(TIMER_PATH)\
		 -I$(RADIX_SORT_PATH) \
		 -I$(RANDOM_PATH) \
		 -I$(CUDA_INCLUDE) \
		 -I$(SDK_SHARED_INC) \
		 -I$(SDK_COMMON_INC) 

LIB_OBJS=$(BSEARCH_PATH)/bsearch.o \
		 $(BSEARCH_PATH)/bsearch_cuda.cu.o \
		 $(MT_PATH)/mt.o \
		 $(TIMER_PATH)/timer.o \
		 $(RADIX_SORT_PATH)/radixsort.cu.o \
		 $(RADIX_SORT_PATH)/radixsort.o \
		 $(RANDOM_PATH)/random.cu.o 

C_SRC=$(BSEARCH_PATH)/bsearch.c \
	  $(MT_PATH)/mt.c \
	  $(TIMER_PATH)/timer.c 

.SUFFIXES: .c .cpp .cu .o

EXECS=b_search \
	  sort_b_search \
	  i_gm_b_search \
	  i_sm_b_search \
	  t_gm_b_search \
	  t_sm_b_search \
	  b_search_correctness \
	  b_search_index_tree_check \
	  sort_i_gm_b_search \
	  sort_t_gm_b_search

all: $(EXECS)

#sort_b_search: sort_b_search.cu.o $(LIB_OBJS)
	#$(CPP) $(LFLAGS_A) -o $@ \
		#$(C_SRC) \
		#$(INCLUDES) \
		#$(LIB_OBJS) \
		#sort_b_search.cu.o \
		#$(LFLAGS_B)

%: %.cu.o $(LIB_OBJS)
	$(CPP) $(LFLAGS_A) -o $@ \
		$(C_SRC) $(INCLUDES) $+ $(LFLAGS_B)

%.cu.o: %.cu
	$(NVCC) $(INCLUDES) -c $< -o $@
	
%.o: %.cpp
	$(CPP) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(EXECS) *.a *.o $(LIB_OBJS)
