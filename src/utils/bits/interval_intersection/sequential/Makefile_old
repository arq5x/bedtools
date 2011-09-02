CFLAGS=-g
CC=/usr/bin/gcc
SRC_PATH=$(HOME)/src
BSEARCH_PATH=$(SRC_PATH)/binary_search/lib
MT_PATH=$(SRC_PATH)/utils/c/mt
TIMER_PATH=$(SRC_PATH)/utils/c/timer

INCLUDES=-I../lib \
		 -I$(BSEARCH_PATH)\
		 -I$(MT_PATH)\
		 -I$(TIMER_PATH)

EXECS=count_interval_intersections_seq \
	  enumerate_interval_intersections_seq \
	  per_interval_count_interval_intersections_seq

LIB_OBJS=../lib/interval.o \
		 $(BSEARCH_PATH)/bsearch.o \
		 $(MT_PATH)/mt.o \
		 $(TIMER_PATH)/timer.o 

all: $(EXECS)

count_interval_intersections_seq: count_interval_intersections_seq.o $(LIB_OBJS)
	$(CC) -o count_interval_intersections_seq \
		count_interval_intersections_seq.o \
		$(LIB_OBJS) \
		-lm

enumerate_interval_intersections_seq: enumerate_interval_intersections_seq.o \
	$(LIB_OBJS)
	$(CC) -o enumerate_interval_intersections_seq \
		enumerate_interval_intersections_seq.o \
		$(LIB_OBJS) \
		-lm

per_interval_count_interval_intersections_seq: \
	per_interval_count_interval_intersections_seq.o \
	$(LIB_OBJS)
	$(CC) -o per_interval_count_interval_intersections_seq \
		per_interval_count_interval_intersections_seq.o \
		$(LIB_OBJS) \
		-lm

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(EXECS) *.a *.o $(LIB_OBJS)

