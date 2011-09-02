#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bsearch.h"

int main(int argc, char *argv[])
{

	if (argc < 4) {
		fprintf(stderr, "usages:\t%s <D size > <Q size> <I/T size> "
				"<seed>\n", argv[0]);
		return 1;
	}
	int D_size = atoi(argv[1]); // size of data set D
	int Q_size = atoi(argv[2]); // size of query set Q
	int I_size = atoi(argv[3]);
	int T_size = I_size;
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
	int *T = (int *) malloc(T_size * sizeof(int));
	create_tree(D, D_size, T, T_size);
	stop();
	unsigned long tree_time = report();

	start();
	int *I = (int *) malloc(I_size * sizeof(int));
	create_index(D, D_size, I, I_size);
	stop();
	unsigned long index_time = report();

	int *BR = (int *) malloc(Q_size * sizeof(int));
	int *IR = (int *) malloc(Q_size * sizeof(int));
	int *TR = (int *) malloc(Q_size * sizeof(int));
	/* Search list */
	unsigned long total_time = 0;
	int j;
	for (j = 0; j < Q_size; j++) {
		BR[j] = bsearch_seq(Q[j], D, D_size, -1, D_size);
		IR[j] = i_bsearch_seq(Q[j], D, D_size, I, I_size);
		TR[j] = t_bsearch_seq(Q[j], D, D_size, T, T_size);
	}

	for (j = 0; j < Q_size; j++)
		if ( (BR[j] != IR[j]) ||
			 (BR[j] != TR[j]) )
			printf("%d\tb:%d\ti:%d\tt:%d\n", j, BR[j], IR[j], TR[j]);
	return 0;
}
