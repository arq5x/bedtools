#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "bsearch.h"
#include "mt.h"

int main(int argc, char *argv[])
{
	MPI_Status status;
	int rank, size, seen;

	MPI_Init (&argc, &argv);    /* starts MPI */
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
	MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */

	if (argc < 4) {
		fprintf(stderr, "usages:\t%s <D size > <Q size> "
				"<seed>\n", argv[0]);
		return 1;
	}
	int D_size = atoi(argv[1]); // size of data set D
	int Q_size = atoi(argv[2]); // size of query set Q
	int seed = atoi(argv[3]);

	/* 
	 * The master thread (0) will create D and Q, send it out to the threads,
	 * then collect the results.  There are two ways to do this, send out
	 * portions of D and all of Q, or portions of Q and all of D, it will
	 * likely depend on the size of D and Q.  We will try both.
	 *
	 * To aggregate the results, we can also either do a reduction or just have
	 * the master calculate everything.  We will try both.
	 */
	if (rank == 0) {
		unsigned int *D = (unsigned int *)
			malloc(D_size * sizeof(unsigned int));
		unsigned int *Q = (unsigned int *)
			malloc(Q_size * sizeof(unsigned int));

		unsigned int *R = (unsigned int *)
			malloc(Q_size * sizeof(unsigned int));

		init_genrand(seed);

		// Have the mater generate all of the data, in other cases we can have
		// the worker nodes read in their own portion of files
		generate_rand_unsigned_int_set(D, D_size);
		generate_rand_unsigned_int_set(Q, Q_size);

		// Let each thread sort their own
		//qsort(D, D_size, sizeof(unsigned int), compare_unsigned_int);

		// In this case we are going to send out all of Q and portions of D.
		//
	}

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
