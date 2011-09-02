#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bsearch.h"

int main(int argc, char *argv[])
{

	if (argc < 5) {
		fprintf(stderr, "usages:\t%s <D size > <Q size> <I size> "
				"<seed>\n", argv[0]);
		return 1;
	}
	int D_size = atoi(argv[1]); // size of data set D
	int Q_size = atoi(argv[2]); // size of query set Q
	int T_size = atoi(argv[3]); // size of index I
	int seed = atoi(argv[4]);

	unsigned int *D = (unsigned int *)
		malloc(D_size * sizeof(unsigned int));
	unsigned int *Q = (unsigned int *)
		malloc(Q_size * sizeof(unsigned int));
	srand(seed);
	generate_rand_unsigned_int_set(D, D_size);
	generate_rand_unsigned_int_set(Q, Q_size);

	qsort(D, D_size, sizeof(unsigned int), compare_unsigned_int);

	start();
	unsigned int *T = (unsigned int *)
		malloc(T_size * sizeof(unsigned int));
	create_tree(D, D_size, T, T_size);
	stop();
	unsigned long tree_time = report();

	unsigned long total_time_1 = 0;
	int j;
	start();
	for (j = 0; j < Q_size; j++) {
		int c = t_bsearch_seq(Q[j], D, D_size, T, T_size);
	}
	stop();
	total_time_1 = report();

	unsigned long total_time_2 = 0;
	start();
	for (j = 0; j < Q_size; j++) {
		int c = t_bsearch_seq(Q[j], D, D_size, T, T_size);
	}
	stop();
	total_time_2 = report();


	printf("%lu,%lu\n", total_time_1 + tree_time,total_time_2 + tree_time);

	return 0;
}
