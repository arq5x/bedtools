CFLAGS=-g
FASTA_PATH=/home/rl6sf/src/utils/c/fasta
CC=g++

INCLUDES=-I$(FASTA_PATH)

EXECS=get_fasta_interval

LIB_OBJS=$(FASTA_PATH)/fasta_file.o

all: $(EXECS)

get_fasta_interval: get_fasta_interval.o $(LIB_OBJS)
	$(CC) -o $@ $+ -lm

%.o: %.cpp
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f $(EXECS) *.a *.o $(LIB_OBJS)

