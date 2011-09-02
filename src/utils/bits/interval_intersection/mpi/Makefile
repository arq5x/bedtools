CFLAGS=
MPICC=/usr/bin/mpicc
SRC_PATH=$(HOME)/src
BSEARCH_PATH=$(SRC_PATH)/binary_search/lib
MT_PATH=$(SRC_PATH)/utils/c/mt
TIMER_PATH=$(SRC_PATH)/utils/c/timer

INCLUDES=-I../lib \
		 -I$(BSEARCH_PATH)\
		 -I$(MT_PATH)\
		 -I$(TIMER_PATH)

EXECS=count_interval_intersections_mpi \
	  per_interval_count_interval_intersections_mpi

LIB_OBJS=../lib/interval.o \
		 $(BSEARCH_PATH)/bsearch.o \
		 $(MT_PATH)/mt.o \
		 $(TIMER_PATH)/timer.o 

.SUFFIXES: .c .cpp .cu .o .mpi.c .mpi.o

all: $(EXECS)

%: %.mpi.o $(LIB_OBJS)
	$(MPICC) $(LFLAGS_A) -o $@ \
		$(C_SRC) $(INCLUDES) $+ $(LFLAGS_B)


#per_interval_count_interval_intersections_mpi: per_interval_count_interval_intersections_mpi.mpi.o \
		$(LIB_OBJS)
	#$(MPICC) -o per_interval_count_interval_intersections_mpi \
		#per_interval_count_interval_intersections_mpi.mpi.o \
		#$(LIB_OBJS) \
		#-lm

#count_interval_intersections_mpi: count_interval_intersections_mpi.mpi.o \
		#$(LIB_OBJS)
	#$(MPICC) -o count_interval_intersections_mpi \
		#count_interval_intersections_mpi.mpi.o \
		#$(LIB_OBJS) \
		#-lm

%.mpi.o: %.mpi.c
	$(MPICC) -c $(CFLAGS) $(INCLUDES) $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(EXECS) *.a *.o $(LIB_OBJS)

