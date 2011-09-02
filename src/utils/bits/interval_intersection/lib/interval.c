#include <stdlib.h>
#include <stdio.h>
#include "interval.h"
#include "bsearch.h"
#include "mt.h"

//{{{ int compare_interval_by_start (const void *a, const void *b)
int compare_interval_by_start (const void *a, const void *b)
{  
	struct interval *a_i = (struct interval *)a;
	struct interval *b_i = (struct interval *)b;
	if (a_i->start < b_i->start)
		return -1;
	else if (a_i->start > b_i->start)
		return 1;
	else
		return 0;
}
//}}}

//{{{ void enumerate_intersections_bsearch_seq(struct interval *A,

/**
  * @param A intervals in set A
  * @param size_A size of set A
  * @param B intervals in set B
  * @param size_B size of set B
  * @param R prefix sum of the intersection between A and B, A[0] is the number
  * of intervals in B that intersect A[0], A[1] is A[0] + the number of
  * intervals in B that intersect A[1], and so on
  * @param E array that will hold the enumberated interval intersections
  * @param size_E 
  */
void enumerate_intersections_bsearch_seq(struct interval *A,
										 unsigned int size_A,
										 struct interval *B,
										 unsigned int size_B,
										 unsigned int *R,
										 unsigned int *E,
										 unsigned int size_E)
{

	unsigned int i, O = 0;

	qsort(B, size_B, sizeof(struct interval), compare_interval_by_start); 

	unsigned int *B_starts =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));

	for (i = 0; i < size_B; i++)
		B_starts[i] = B[i].start;

	unsigned int start = 0, end = 0, o;

	for (i = 0; i < size_A; i++) {
		end = R[i];
		o = end - start;

		if (o > 0) {
			//printf("%d\t", i);
			unsigned int from = bsearch_seq(A[i].end,
										B_starts,
										size_B,
										-1,
										size_B);

			while ( ( B_starts[from] == A[i].end) && from < size_B)
				++from;

			while ( (from > 0) && (o > 0) ) {
				// test if A[i] intersects B[from]
				if ( (A[i].start <= B[from].end) &&
					 (A[i].end >= B[from].start) ) {
					E[start] = from;
					//printf("%d,%u\t", start,from);
					--o;
					start++;
				} 

				--from;
			}
			//printf("\n");
		}
	}
}
//}}}

//{{{ unsigned int per_interval_count_intersections_bsearch_seq(struct interval *A,
unsigned int per_interval_count_intersections_bsearch_seq(struct interval *A,
														   unsigned int size_A,
														   struct interval *B,
														   unsigned int size_B,
														   unsigned int *R)
{

	unsigned int i, O = 0;

	unsigned int *B_starts =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));
	unsigned int *B_ends =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));

	for (i = 0; i < size_B; i++) {
		B_starts[i] = B[i].start;
		B_ends[i] = B[i].end;
	}

	qsort(B_starts, size_B, sizeof(unsigned int), compare_unsigned_int); 
	qsort(B_ends, size_B, sizeof(unsigned int), compare_unsigned_int); 

	for (i = 0; i < size_A; i++) {
		unsigned int num_cant_before = bsearch_seq(A[i].start,
												   B_ends,
												   size_B,
												   -1,
												   size_B);
		unsigned int b = bsearch_seq(A[i].end,
								     B_starts,
								     size_B,
								     -1,
								     size_B);

		while ( ( B_starts[b] == A[i].end) && b < size_B)
			++b;

		unsigned int num_cant_after = size_B - b;

		unsigned int num_left = size_B - num_cant_before - num_cant_after;
		O += num_left;
		R[i] = num_left;
	}

	return O;
}
//}}}

//{{{ unsigned int count_intersections_bsearch_seq(struct interval *A,
unsigned int count_intersections_bsearch_seq(struct interval *A,
										     unsigned int size_A,
											 struct interval *B,
										     unsigned int size_B)
{

	unsigned int i, O = 0;

	unsigned int *B_starts =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));
	unsigned int *B_ends =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));

	for (i = 0; i < size_B; i++) {
		B_starts[i] = B[i].start;
		B_ends[i] = B[i].end;
	}

	qsort(B_starts, size_B, sizeof(unsigned int), compare_unsigned_int); 
	qsort(B_ends, size_B, sizeof(unsigned int), compare_unsigned_int); 

	for (i = 0; i < size_A; i++) {
		unsigned int num_cant_before = bsearch_seq(A[i].start,
												   B_ends,
												   size_B,
												   -1,
												   size_B);
		unsigned int b = bsearch_seq(A[i].end,
								     B_starts,
								     size_B,
								     -1,
								     size_B);

		while ( ( B_starts[b] == A[i].end) && b < size_B)
			++b;

		unsigned int num_cant_after = size_B - b;

		unsigned int num_left = size_B - num_cant_before - num_cant_after;
		O += num_left;
	}

	return O;
}
//}}}

//{{{ unsigned int count_intersections_i_bsearch_seq(struct interval *A,
unsigned int count_intersections_i_bsearch_seq(struct interval *A,
										       unsigned int size_A,
											   struct interval *B,
										       unsigned int size_B,
											   unsigned int size_I)
{

	unsigned int i, O = 0;

	unsigned int *B_starts =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));
	unsigned int *B_ends =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));

	for (i = 0; i < size_B; i++) {
		B_starts[i] = B[i].start;
		B_ends[i] = B[i].end;
	}

	qsort(B_starts, size_B, sizeof(unsigned int), compare_unsigned_int); 
	qsort(B_ends, size_B, sizeof(unsigned int), compare_unsigned_int); 

	unsigned int *I_starts = (unsigned int *)
			malloc(size_I * sizeof(unsigned int));
	unsigned int *I_ends = (unsigned int *)
			malloc(size_I * sizeof(unsigned int));

	create_index(B_starts, size_B, I_starts, size_I);
	create_index(B_ends, size_B, I_ends, size_I);

	for (i = 0; i < size_A; i++) {
		unsigned int num_cant_before = i_bsearch_seq(A[i].start,
												B_ends,
												size_B,
												I_ends,
												size_I);
		unsigned int b = i_bsearch_seq(A[i].end,
									   B_starts,
									   size_B,
									   I_starts,
									   size_I);

		//nsigned int x = b;

		while ( ( B_starts[b] == A[i].end) && b < size_B)
			++b;

		unsigned int num_cant_after = size_B - b;

		unsigned int num_left = size_B - num_cant_before - num_cant_after;

		O += num_left;

		/*
		printf("i\t"
			   "%u\t"
			   "%u\t"
			   "%u\t"
			   "%u\t"
			   "%u\n",
			   num_cant_before,
			   num_cant_after,
			   A[i].end,
			   B_starts[x],
			   O);
		*/
	}

	return O;
}
//}}}

//{{{ unsigned int count_intersections_t_bsearch_seq(struct interval *A,
unsigned int count_intersections_t_bsearch_seq(struct interval *A,
										       unsigned int size_A,
											   struct interval *B,
										       unsigned int size_B,
											   unsigned int size_T)
{

	unsigned int i, O = 0;

	unsigned int *B_starts =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));
	unsigned int *B_ends =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));

	for (i = 0; i < size_B; i++) {
		B_starts[i] = B[i].start;
		B_ends[i]   = B[i].end;
	}

	qsort(B_starts, size_B, sizeof(unsigned int), compare_unsigned_int); 
	qsort(B_ends, size_B, sizeof(unsigned int), compare_unsigned_int); 

	unsigned int *T_starts = (unsigned int *)
			malloc(size_T * sizeof(unsigned int));
	unsigned int *T_ends = (unsigned int *)
			malloc(size_T * sizeof(unsigned int));

	create_tree(B_starts, size_B, T_starts, size_T);
	create_tree(B_ends,   size_B, T_ends,   size_T);

	for (i = 0; i < size_A; i++) {

		unsigned int num_cant_before = t_bsearch_seq(A[i].start,
												B_ends,
												size_B,
												T_ends,
												size_T);



		unsigned int b = t_bsearch_seq(A[i].end,
									   B_starts,
									   size_B,
									   T_starts,
									   size_T);

		//unsigned int x = b;

		while ( ( B_starts[b] == A[i].end) && b < size_B)
			++b;

		unsigned int num_cant_after = size_B - b;

		unsigned int num_left = size_B - num_cant_before - num_cant_after;

		O += num_left;

		/*
		printf("t\t"
			   "%u\t"
			   "%u\t"
			   "%u\t"
			   "%u\t"
			   "%u\n",
			   num_cant_before,
			   num_cant_after,
			   A[i].end,
			   B_starts[x],
			   O);
		*/
	}

	return O;
}
//}}}

//{{{ unsigned int count_intersections_bsearch_seq(struct interval *A,
unsigned int count_intersections_sort_bsearch_seq(struct interval *A,
										          unsigned int size_A,
											      struct interval *B,
										          unsigned int size_B)
{

	unsigned int i, O = 0;

	unsigned int *B_starts =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));
	unsigned int *B_ends =
			(unsigned int *) malloc(size_B * sizeof(unsigned int));

	for (i = 0; i < size_B; i++) {
		B_starts[i] = B[i].start;
		B_ends[i] = B[i].end;
	}

	qsort(B_starts, size_B, sizeof(unsigned int), compare_unsigned_int); 
	qsort(B_ends, size_B, sizeof(unsigned int), compare_unsigned_int); 

	qsort(A, size_A, sizeof(struct interval), compare_interval_by_start); 

	for (i = 0; i < size_A; i++) {
		unsigned int num_cant_before = bsearch_seq(A[i].start,
												   B_ends,
												   size_B,
												   -1,
												   size_B);
		unsigned int b = bsearch_seq(A[i].end,
								     B_starts,
								     size_B,
								     -1,
								     size_B);

		while ( ( B_starts[b] == A[i].end) && b < size_B)
			++b;

		unsigned int num_cant_after = size_B - b;

		unsigned int num_left = size_B - num_cant_before - num_cant_after;
		O += num_left;
	}

	return O;
}
//}}}

//{{{unsigned int count_intersections_brute_force_seq(struct interval *A,
unsigned int count_intersections_brute_force_seq(struct interval *A,
												 unsigned int size_A,
											     struct interval *B,
										         unsigned int size_B)
{
	unsigned int i, j, O = 0;

	for (i = 0; i < size_A; i++) 
		for (j = 0; j < size_B; j++) 
			if ( ( A[i].start <= B[j].end ) &&
				 ( A[i].end >= B[j].start ) )
				++O;
	return O;
}
//}}}

//{{{void generate_interval_sets(struct interval *A,
// must run init_genrand(seed) first
void generate_interval_sets(struct interval *A,
							unsigned int size_A,
							unsigned int len_A,
							struct interval *B,
							unsigned int size_B,
							unsigned int len_B,
							unsigned int P)
{

	int i;

    for (i = 0; i < size_A; i++) {
		A[i].start = genrand_int32();
		A[i].end = A[i].start + len_A;
	}
	
	qsort(A, size_A, sizeof(struct interval), compare_interval_by_start); 
	
	/*
	 * Draw a number to see if the next interval will intersect or not.
	 * Draw a number to get the next interval, make new interval intersect or
	 * not with the drawn interval based on the first draw.
	 */
	int p_max = 100;
	unsigned int p_mask = get_mask(p_max);
	unsigned int i_mask = get_mask(size_A);
	unsigned int l_mask = get_mask(len_A);

    for (i = 0; i < size_B; i++) {
		unsigned int next_i = get_rand(size_A, i_mask);
		unsigned int next_p = get_rand(p_max, p_mask);

		if (P >= next_p) // intersect
			// Pick an rand between start and end to start from
			B[i].start = A[next_i].start + get_rand(len_A, l_mask);
		else  // do not intersect
			B[i].start = A[next_i].end + get_rand(len_A, l_mask);

		B[i].end = B[i].start + len_B;
	}
}
//}}}

//{{{void generate_interval_sets(struct interval *A,
// must run init_genrand(seed) first
void generate_ind_interval_sets(struct interval *A,
							unsigned int size_A,
							unsigned int len_A,
							struct interval *B,
							unsigned int size_B,
							unsigned int len_B)
{

	int i;

    for (i = 0; i < size_A; i++) {
		A[i].start = genrand_int32();
		A[i].end = A[i].start + len_A;
	}
	
    for (i = 0; i < size_B; i++) {
		B[i].start = genrand_int32();
		B[i].end = B[i].start + len_B;
	}
}
//}}}
