#include "bits.h"
#include <stdio.h>

//{{{int b_search(int key, int *D, int D_size, int lo, int hi)
uint32_t uint32_bsearch(uint32_t key, uint32_t *D, uint32_t D_size, int lo, uint32_t hi)
{
	//int lo = -1, hi = D_size, mid;
	uint32_t i = 0;
	int mid;
	//fprintf(stderr, "k:%d\t", key);
	while ( hi - lo > 1) {
		++i;
		mid = (hi + lo) / 2;
		//printf("%d\t", D[mid]);
		//fprintf(stderr, "%d,%d,%d\t", lo, mid, hi, D[mid]);
		if ( D[mid] < key )
			lo = mid;
		else
			hi = mid;
	}
	//fprintf(stderr, "\n");
	
	return hi;
}
//}}}


//{{{ int count_intersections_bsearch_seq( struct interval *A_r,
uint32_t count_intersections_bsearch_seq(uint32_t *A_starts,
                                    uint32_t *A_ends, 
									uint32_t A_size,
									uint32_t *B_starts,
									uint32_t *B_ends,  
									uint32_t B_size )
{
	uint32_t i, O = 0;

	for (i = 0; i < A_size; i++) {
		// Find the position of the last interval to END before the query
		// interval STARTS
		uint32_t num_cant_before = uint32_bsearch(A_starts[i], B_ends, B_size, -1, B_size);
		// The key in this search is the start of the query, and the search
		// space is the set of interval ends.
		// The result, a,  is either the 
		// (1) insert position:
		//		the seiries of  intervals A_end[0], ..., A_end[a-1] end before
		//		the query starts, and the intervals 
		//		A_end[a], ..., A_end[A_size - 1] end after the query starts.
		//		In this case a is also the size of the set 
		//		{A_end[0], ... ,A_end[a-1]}
		// (2) position of a match:
		//		since we are talking about closed intervals, the interval 
		//		A_end[a] does not end before the query starts, so 
		//		the seiries of intervals the end before the query starts is 
		//		is A_end[0], ..., A_end[a-1], and a is still the size of that
		//		set

		// Find the position of the first interval to START after the query
		// ENDS
		uint32_t b = uint32_bsearch(A_ends[i], B_starts, B_size, -1, B_size);
		// The key in this search is the end of the query, and the search space
		// is the set of interval starts
		// The result, b, is either the
		// (1) insert position:
		//		in this case, A[b] starts after the query ends, so the series of
		//		intervals A[b], ..., A[A_size - 1] start after the querty and,
		//		and the size of that set is A_size - b
		// (2) position of a match:
		//		int this case, A[b + 1] starts after the query ends, and the
		//		size of that set is A_size - b - 1.  It is possible  that
		//		A[b] == A[b + 1] == A[b + 2] and so on, in which case we need a
		//		little while loop to move b past all of these

		//int start_pos = MAX(0,a);
		//int end_pos = MIN(A_size - 1,b);


		/*
		while ( ( A_end[a].end == B[i].start ) && (a > 0))
			--a;
		*/

		while ( ( B_starts[b] == A_ends[i]) && b < B_size)
			++b;

		uint32_t num_cant_after = B_size - b;


		uint32_t num_left = B_size - num_cant_before - num_cant_after;
		O += num_left;
	}

	return O;
}
// }}}