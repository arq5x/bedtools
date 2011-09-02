#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mt.h"
#include "timer.h"
#include "interval.h"

int main(int argc, char *argv[])
{

	if (argc < 8) {
		fprintf(stderr, "usage:\t%s <size A> <len A> <size B> <len B> "
						"<seed> <P> <size I>\n",
				argv[0]);
		return 1;
	}

	unsigned int size_A = atoi(argv[1]);
	unsigned int len_A = atoi(argv[2]);
	unsigned int size_B = atoi(argv[3]);
	unsigned int len_B = atoi(argv[4]);
	unsigned int seed = atoi(argv[5]);
	unsigned int P = atoi(argv[6]);
	unsigned int size_I = atoi(argv[7]);
	unsigned int size_T = size_I;

	if (size_I > size_B) {
		fprintf(stderr, "Index larger than DB.\n");
		return 1;
	}

	struct interval *A = (struct interval *)
			malloc(size_A * sizeof(struct interval));
	struct interval *B = (struct interval *)
			malloc(size_B * sizeof(struct interval));


	init_genrand(seed);
	generate_interval_sets(A, size_A, len_A, B, size_B, len_B, P);

	start();
	unsigned int O = count_intersections_bsearch_seq(A, size_A, B, size_B);
	stop();
	unsigned long bsearch_time = report();

	start();
	unsigned int OI = count_intersections_i_bsearch_seq(A, size_A, B, size_B,
			size_I);
	stop();
	unsigned long i_bsearch_time = report();

	start();
	unsigned int OT = count_intersections_t_bsearch_seq(A, size_A, B, size_B,
			size_T);
	stop();
	unsigned long t_bsearch_time = report();

	start();
	unsigned int OS = count_intersections_sort_bsearch_seq(A, size_A, B,
			size_B);
	stop();
	unsigned long sort_bsearch_time = report();

	/*
	start();
	unsigned int OB = count_intersections_brute_force_seq(A, size_A, B, size_B);
	stop();
	unsigned long brute_force_time = report();
	*/

	printf("b:%u,%lu\n"
		   "i:%u,%lu\n"
		   "t:%u,%lu\n"
		   "s:%u,%lu\n",
		   O, bsearch_time,
		   OI, i_bsearch_time,
		   OT, t_bsearch_time,
		   OS, sort_bsearch_time);
}
