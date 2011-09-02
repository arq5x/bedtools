#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bsearch.h"
#include "mt.h"

int main(int argc, char *argv[])
{

	if (argc < 4) {
		fprintf(stderr, "usages:\t%s <D size > <Q size> "
				"<seed>\n", argv[0]);
		return 1;
	}
	int D_size = atoi(argv[1]); // size of data set D
	int Q_size = atoi(argv[2]); // size of query set Q
	int seed = atoi(argv[3]);

	unsigned int *D = (unsigned int *)
		malloc(D_size * sizeof(unsigned int));
	unsigned int *Q = (unsigned int *)
		malloc(Q_size * sizeof(unsigned int));
	//srand(seed);
	init_genrand(seed);

	generate_rand_unsigned_int_set(D, D_size);
	generate_rand_unsigned_int_set(Q, Q_size);

	qsort(D, D_size, sizeof(unsigned int), compare_unsigned_int);

	unsigned long total_time_1 = 0;
	int j;
	start();
	for (j = 0; j < Q_size; j++)  {
		int c = bsearch_seq(Q[j], D, D_size, -1, D_size);
	}
	stop();
	total_time_1 = report();

	unsigned long total_time_2 = 0;
	start();
	for (j = 0; j < Q_size; j++) {
		int c = bsearch_seq(Q[j], D, D_size, -1, D_size);
	}
	stop();
	total_time_2 += report();

	printf("%lu,%lu\n", total_time_1, total_time_2);

	return 0;
}
