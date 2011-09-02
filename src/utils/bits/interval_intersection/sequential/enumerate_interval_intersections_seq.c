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

	unsigned int *R = (unsigned int *)
			malloc(size_A * sizeof(unsigned int));


	init_genrand(seed);
	generate_interval_sets(A, size_A, len_A, B, size_B, len_B, P);

	start();
	unsigned int O = per_interval_count_intersections_bsearch_seq(
		A, size_A, B, size_B, R);

	unsigned int T = 0;
	int i;
	for (i = 0; i < size_A; i++) {
		//printf("%d\ti:%u\n", i, R[i]);
		T = T + R[i];
		R[i] = T;
	}
	
	unsigned int *E = (unsigned int *)
			malloc(O * sizeof(unsigned int));

	enumerate_intersections_bsearch_seq(A,
											 size_A,
											 B,
											 size_B,
											 R,
											 E,
											 O);
	stop();
	unsigned long bsearch_time = report();

	unsigned int start = 0, end = 0;
	for (i = 0; i < size_A; i++) {
		end = R[i];
		printf("%d (%u): ", i, end-start);
		for ( ; start < end; start++)
			printf("%u\t", E[start]);
		printf("\n");
	}


	printf("b:%u,%lu\n",
		   O, bsearch_time);
	for (i = 0; i < size_A; i++) 
		printf("%d\t(%u,%u)\t(%u,%u)\n",
				i,
			A[i].start,
			A[i].end,
			B[i].start,
			B[i].end);
}
