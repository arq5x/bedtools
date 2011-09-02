CFLAGS=-g
MT_PATH=/home/rl6sf/src/utils/c/mt
TIMER_PATH=/home/rl6sf/src/utils/c/timer
INCLUDES=-I../lib \
		 -I$(MT_PATH) \
		 -I$(TIMER_PATH)
LIB_OBJS=$(TIMER_PATH)/timer.o \
		 $(MT_PATH)/mt.o \
		 ../lib/bsearch.o

EXECS=i_bsearch \
	  t_bsearch \
	  bsearch \
	  sort_bsearch \
	  bsearch_correctness

all: $(EXECS)

i_bsearch: i_bsearch.o $(LIB_OBJS)
	gcc -o i_bsearch i_bsearch.o $(LIB_OBJS) -lm

t_bsearch: t_bsearch.o $(LIB_OBJS)
	gcc $(CFLAGS) -o t_bsearch t_bsearch.o $(LIB_OBJS) -lm

bsearch: bsearch.o $(LIB_OBJS)
	gcc -o bsearch bsearch.o $(LIB_OBJS) -lm

sort_bsearch: sort_bsearch.o $(LIB_OBJS)
	gcc -o sort_bsearch sort_bsearch.o $(LIB_OBJS) -lm

bsearch_correctness: bsearch_correctness.o $(LIB_OBJS)
	gcc -o bsearch_correctness bsearch_correctness.o $(LIB_OBJS) -lm

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(EXECS) *.a *.o $(LIB_OBJS)

