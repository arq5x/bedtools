#include "../lib/bed.h"
#include "../lib/set_intersect.h"

//{{{ compare_*
int compare_interval_pairs_by_start (const void *a, const void *b)
{
	struct interval_pair *a_i = (struct interval_pair *)a;
	struct interval_pair *b_i = (struct interval_pair *)b;
	if (a_i->start < b_i->start)
		return -1;
	else if (a_i->start > b_i->start)
		return 1;
	else
		return 0;

}

int compare_interval_triples_by_start (const void *a, const void *b)
{
	struct interval_triple *a_i = (struct interval_triple *)a;
	struct interval_triple *b_i = (struct interval_triple *)b;
	if (a_i->start < b_i->start)
		return -1;
	else if (a_i->start > b_i->start)
		return 1;
	else
		return 0;
}

int compare_interval_triples_start_to_end (const void *key, const void *b)
{
	struct interval_triple *key_i = (struct interval_triple *)key;
	struct interval_triple *b_i = (struct interval_triple *)b;
	return key_i->start - b_i->end;
}


int compare_interval_triples_by_end (const void *a, const void *b)
{
	struct interval_triple *a_i = (struct interval_triple *)a;
	struct interval_triple *b_i = (struct interval_triple *)b;
	if (a_i->end < b_i->end)
		return -1;
	else if (a_i->end > b_i->end)
		return 1;
	else
		return 0;
}

int compare_triple_lists (const void *a, const void *b)
{
	struct triple *a_i = (struct triple *)a;
	struct triple *b_i = (struct triple *)b;
	//return a_i->key - b_i->key;
	if (a_i->key < b_i->key)
		return -1;
	else if (a_i->key > b_i->key)
		return 1;
	else if (a_i->type < b_i->type)
		return -1;
	else if (a_i->type > b_i->type)
		return 1;
	else if (a_i->sample < b_i->sample)
		return -1;
	else if (a_i->sample > b_i->sample)
		return 1;
	else
		return 0;
}

int compare_pair_lists (const void *a, const void *b)
{
	struct pair *a_i = (struct pair *)a;
	struct pair *b_i = (struct pair *)b;
	return a_i->key - b_i->key;
}

int compare_uints (const void *a, const void *b)
{
	unsigned int *a_i = (unsigned int *)a;
	unsigned int *b_i = (unsigned int *)b;
	if (*a_i < *b_i)
		return -1;
	else if (*a_i > *b_i)
		return 1;
	else
		return 0;
}
//}}}

//{{{ void set_start_len( struct bed_line *U_array,
/*
 * @param U_array the universe to consider
 * @param U_size size of the universe
 * @param A_array set of intervals to map
 * @param A_start set of resulting start positions
 * @param A_len set of resulting lengths
 * @param A_size number of elements to map
 */
void set_start_len( struct bed_line *U_array,
					int U_size,
					struct bed_line *A_array,
					unsigned int *A_start,
					unsigned int *A_len,
					int A_size )
{
	int i, j, k = 0;
	for (i = 0; i < A_size; i++) {
		int start = -1, offset = -1;
		for (j = 0; j < U_size; j++) {
			if ( ( U_array[j].chr == A_array[i].chr) &&
				 ( U_array[j].start <= A_array[i].end) &&
				 ( U_array[j].end >= A_array[i].start) ) {

				start = U_array[j].start;
				offset = U_array[j].offset;
				break;
			}
		}
		A_start[k] = A_array[i].start - start + offset;
		A_len[k] = A_array[i].end -A_array[i].start;
		++k;
	}
}
//}}}

//{{{ void set_start_end( struct bed_line *U_array,
/*
 * @param U_array the universe to consider
 * @param U_size size of the universe
 * @param A_array set of intervals to map
 * @param A_start set of resulting start positions
 * @param A_end set of resulting lengths
 * @param A_size number of elements to map
 */
void set_start_end( struct bed_line *U_array,
					int U_size,
					struct bed_line *A_array,
					unsigned int *A_start,
					unsigned int *A_end,
					int A_size )
{
	int i, j, k = 0;
	for (i = 0; i < A_size; i++) {
		int start = -1, offset = -1;
		for (j = 0; j < U_size; j++) {
			if ( ( U_array[j].chr == A_array[i].chr) &&
				 ( U_array[j].start <= A_array[i].end) &&
				 ( U_array[j].end >= A_array[i].start) ) {

				start = U_array[j].start;
				offset = U_array[j].offset;
				break;
			}
		}
		A_start[k] = A_array[i].start - start + offset;
		A_end[k] = A_array[i].end - start + offset;
		++k;
	}
}
//}}}

//{{{ int enumerate_intersections_bsearch( struct interval *A_r,
int enumerate_intersections_bsearch( unsigned int *A_start,
								 unsigned int *A_len,
								 int A_size,
								 unsigned int *B_start,
								 unsigned int *B_len,
								 int B_size,
								 unsigned int *pairs)
{
	int i, c = 0;
	for (i = 0; i < A_size; i++) {
		// Search for the left-most interval in B with the start in A
		int lo = -1, hi = B_size, mid;
		while ( hi - lo > 1) {
			mid = (hi + lo) / 2;

			if ( B_start[mid] < A_start[i] ) 
				lo = mid;
			else
				hi = mid;

		}

		int left = hi;

		lo = -1;
		hi = B_size;
		while ( hi - lo > 1) {
			mid = (hi + lo) / 2;

			if ( B_start[mid] < (A_start[i] + A_len[i]) ) 
				lo = mid;
			else
				hi = mid;
		}

		int right = hi;

		//printf("i:%d\tl:%d\tr:%d\n", i, left, right);

		/* This is the way to save the intersecting pairs

		// Check to see if the start is in an interval
		int first_hit = 0;
		if ( ( A_start[i] == B_start[left] ) ) {

			++c;

			printf("%d (%u,%u)\t%d (%u,%u) %d\n",
					i, A_start[i], A_start[i] + A_len[i],
					left, B_start[left], B_start[left] + B_len[left],
					c);
			first_hit = 1;

		} else if ( ( left > 0 ) && 
					(A_start[i] <= B_start[left - 1] + B_len[left - 1] ))   {
			++c;

			printf("%d (%u,%u)\t%d (%u,%u) %d\n",
					i, A_start[i], A_start[i] + A_len[i],
					left - 1, B_start[left - 1], 
					B_start[left - 1] + B_len[left - 1],
					c);
		}

		// Check to see if the end is in an interval
		int k;
		for (k = left + first_hit; (k <= right) && (k < B_size); k++) {
			if ( (A_start[i] + A_len[i] >= B_start[k]) ) 
				++c;

				printf("%d (%u,%u)\t%d (%u,%u) %d\n",
						i, A_start[i], A_start[i] + A_len[i],
						k, B_start[k], B_start[k] + B_len[k],
						c);
		}
		*/

		/* v1
		*/
		int range_start, range_end;

		if ( ( A_start[i] == B_start[left] ) ) {
			range_start = left;
		} else if ( ( left > 0 ) &&
					( A_start[i] <= B_start[left - 1] + B_len[left - 1]) ) {
			range_start = left - 1;
		} else {
			range_start = left;
		}


		if ( (right < B_size) &&
			 ( A_start[i] + A_len[i] == B_start[right] ) ) {
			range_end = right;
		} else {
			range_end = right - 1;
		} 

		//c += range_end - range_start + 1;

		int j;
		for (j = range_start; j <= range_end; j++) {
			pairs[2*c] = i;
			pairs[2*c+1] = j;
			++c;
		}
		printf("%d\n", c);
	}

	return c;
}
// }}}

//{{{ int add_offsets( struct chr_list *U_list, 
/*
 * The universe defines the space under consideration.  Each interval in the
 * universe is given an offset in the continguous space.  This offset is used
 * to map intervals in the sample to the continguous space. 
 *
 */
unsigned int add_offsets( struct chr_list *U_list, 
						  int chrom_num )
{
	int i;
	unsigned int c = 0, max = 0;
	for (i = 0; i < chrom_num; i++) {
		struct interval_node *curr = U_list[i].head;
		while (curr != NULL) {
			curr->offset = c;

			unsigned int end = c + curr->end - curr->start;
			if (end > max)
				max = end;

			c += curr->end - curr->start;
			curr = curr->next;
		}
	}

	return max;
}
//}}}

//{{{ int find_intersecting_ranks( struct triple *AB,
/*
 * Requires AB to be sorted
 */
int find_intersecting_ranks( struct triple *AB,
							 int A_size,
							 int B_size,
							 unsigned int *pairs)
{

	int num_pairs = 0;
	int rankA = -1, rankB = -1;
	int inB = 0, inA = 0;
	int i;
	for (i = 0; i < (2*A_size + 2*B_size); i++) {
		if ( AB[i].sample == 1) { //B
			rankB = AB[i].rank;
			inB = !(AB[i].type);
		} else  {//A
			rankA = AB[i].rank;
			inA = !(AB[i].type);
		}

		if (inA && inB) {
			pairs[2*num_pairs] = rankA;
			pairs[2*num_pairs + 1] = rankB;
			++num_pairs;
		}
	}

	return num_pairs;
}
//}}}

//{{{ int check_observed_ranks( int *pairs,
int check_observed_ranks( int *pairs,
						  unsigned int *A_r,
						  unsigned int *A_len,
						  unsigned int *B_r,
						  unsigned int *B_len,
						  int num_pairs,
						  int *R )
{
	int x = 0;
	int rankA, rankB;
	unsigned int A_start, A_end, B_start, B_end;
	int i;
	for (i = 0; i < num_pairs; i++) {
		rankA = pairs[i*2];	
		rankB = pairs[i*2 + 1];	
		A_start = A_r[rankA];
		A_end = A_r[rankA] + A_len[rankA];
		B_start = B_r[rankB];
		B_end = B_r[rankB] + B_len[rankB];

		/**** DEBUG
		fprintf(stderr,"%d,%d,%d\t%d,%d,%d\t%d\n",
				A_start, A_end, A_len[rankA], B_start, B_end, B_len[rankB],
				(A_start <= B_end) && (A_end >= B_start));
		****/

		R[i] += (A_start <= B_end) && (A_end >= B_start);
		x += (A_start <= B_end) && (A_end >= B_start);
	}

	return x;
}
//}}}

//{{{ void map_intervals( struct triple *A, 
/*
 * Each interval becomes a triple: 
 *   key:  offset
 *   sample:  A (0) or B (1)
 *   type:  start (0) or  end (1)
 *   rank: order within
 *
 */
void map_intervals( struct triple *A, 
					struct bed_line *A_array,
					int A_size, 
					struct bed_line *U_array, 
					int U_size,
					int sample)
{
	int i, j, k = 0;
	for (i = 0; i < A_size; i++) {
		unsigned int start = 0, offset = 0;
		// find the universe interval the current interval is in
		// so we can caclulate its place in the continous space
		for (j = 0; j < U_size; j++) {
			if ( ( U_array[j].chr == A_array[i].chr) &&
				 ( U_array[j].start <= A_array[i].end) &&
				 ( U_array[j].end >= A_array[i].start) ) {
				start = U_array[j].start;
				offset = U_array[j].offset;
				break;
			}
		}
		A[k].key = A_array[i].start - start + offset;
		A[k].type = 0;// start
		A[k].sample = sample; // A(0) or B(1)
		++k;
		A[k].key = A_array[i].end - start + offset;
		A[k].type = 1;// end
		A[k].sample = sample; // A(0) or B(1)
		++k;
		/*
		fprintf(stderr, "chr:%d\tstart:%d\tend:%d\tstart:%u\tend:%d\n",
				A_array[i].chr, A_array[i].start, A_array[i].end, 
				A_array[i].start - start + offset,
				A_array[i].end - start + offset);
				*/
	}
}
//}}}

//{{{ void map_to_interval_triple( struct interval_triple *A, 
void map_to_interval_triple( struct interval_triple *A, 
					struct bed_line *A_array,
					int A_size, 
					struct bed_line *U_array, 
					int U_size,
					int sample)
{
	int i, j;
	for (i = 0; i < A_size; i++) {
		unsigned int start = 0, offset = 0;
		// find the universe interval the current interval is in
		// so we can caclulate its place in the continous space
		for (j = 0; j < U_size; j++) {
			if ( ( U_array[j].chr == A_array[i].chr) &&
				 ( U_array[j].start <= A_array[i].end) &&
				 ( U_array[j].end >= A_array[i].start) ) {
				start = U_array[j].start;
				offset = U_array[j].offset;
				//printf("s:%d\to:%d\n", start, offset);
				break;
			}
		}
		A[i].start = A_array[i].start - start + offset;
		A[i].end = A_array[i].end - start + offset;
	}
}
//}}}

//{{{void map_to_interval_pair( struct interval_triple *A, 
void map_to_interval_pair( struct interval_pair *A, 
					struct bed_line *A_array,
					int A_size, 
					struct bed_line *U_array, 
					int U_size,
					int sample)
{
	int i, j;
	for (i = 0; i < A_size; i++) {
		unsigned int start = 0, offset = 0;
		// find the universe interval the current interval is in
		// so we can caclulate its place in the continous space
		for (j = 0; j < U_size; j++) {
			if ( ( U_array[j].chr == A_array[i].chr) &&
				 ( U_array[j].start <= A_array[i].end) &&
				 ( U_array[j].end >= A_array[i].start) ) {
				start = U_array[j].start;
				offset = U_array[j].offset;
				//printf("s:%d\to:%d\n", start, offset);
				break;
			}
		}
		A[i].start = A_array[i].start - start + offset;
		A[i].end = A_array[i].end - start + offset;
	}
}
//}}}

//{{{ void map_to_start_len_array( unsigned int *A_start, 
void map_to_start_len_array( unsigned int *A_start, 
							 unsigned int *A_len, 
							 struct bed_line *A_array,
							 int A_size, 
							 struct bed_line *U_array, 
							 int U_size)
{
	int i, j;
	for (i = 0; i < A_size; i++) {
		unsigned int start = 0, offset = 0;
		// find the universe interval the current interval is in
		// so we can caclulate its place in the continous space
		for (j = 0; j < U_size; j++) {
			if ( ( U_array[j].chr == A_array[i].chr) &&
				 ( U_array[j].start <= A_array[i].end) &&
				 ( U_array[j].end >= A_array[i].start) ) {
				start = U_array[j].start;
				offset = U_array[j].offset;
				break;
			}
		}
		A_start[i] = A_array[i].start - start + offset;
		A_len[i] = A_array[i].end - A_array[i].start;
	}
}
//}}}

//{{{ int count_intersections_sweep_seq
/*
 * AB is a single sorted list with both A and B elements
 */
int count_intersections_sweep_seq ( struct triple *AB, 
									int A_size,
									int B_size )
{
	int num_pairs = 0;
	int inB = 0, inA = 0;
	int i;
	for (i = 0; i < (2*A_size + 2*B_size); i++) {
		if ( AB[i].sample == 1) // B
			if ( AB[i].type == 0 ) { // start
				inB++;
				//num_pairs += inB * inA;
				num_pairs += inA;
			} else
				inB--;
		else // A
			if ( AB[i].type == 0 ) { // start
				inA++;
				//num_pairs += inB * inA;
				num_pairs += inB;
			} else
				inA--;
	}

	return num_pairs;
}
//}}}

//{{{ int count_intersections_brute_force_seq( unsigned int *A, 
/*
 * Scan two sorted lists counting overlaps
 */
int count_intersections_brute_force_seq( struct interval_triple *A_t, 
										 int A_size,
										 struct interval_triple *B_t, 
										 int B_size )
{
	int i, j, O = 0;

	for (i = 0; i < A_size; i++) 
		for (j = 0; j < B_size; j++) 
			if ( ( A_t[i].start <= B_t[j].end ) &&
				 ( A_t[i].end >= B_t[j].start ) )
				++O;
	return O;
}
//}}}

//{{{ int count_intersections_bsearch_seq( struct interval *A_r,
int count_intersections_bsearch_seq(struct interval_triple *A, 
									struct interval_triple *A_end, 
									int A_size,
									struct interval_triple *B, 
									int B_size )
{
	int i, O = 0;

	for (i = 0; i < B_size; i++) {
		// Find the position of the last interval to END before the query
		// interval STARTS
		int a = interval_triple_bsearch_end(A_end, A_size, B[i].start);
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
		int b = interval_triple_bsearch_start(A, A_size, B[i].end);
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

		int num_cant_before = a; 

		while ( ( A[b].start == B[i].end ) && b < A_size)
			++b;

		int num_cant_after = A_size - b;


		int num_left = A_size - num_cant_before - num_cant_after;

		O += num_left;
	}

	return O;
}
// }}}

//{{{ void big_count_intersections_bsearch_seq( struct interval *A_r,
void big_count_intersections_bsearch_seq(unsigned int *A_start, 
										 unsigned int *A_len, 
										 int A_size,
										 unsigned int *B_start, 
										 unsigned int *B_end, 
										 int B_size,
										 unsigned int *R)
{
	int i;

	for (i = 0; i < A_size; i++) {
		unsigned int start = A_start[i];
		unsigned int end = start + A_len[i];

		int a = unsigned_bsearch(B_end, B_size, start);
		int b = unsigned_bsearch(B_start, B_size, end);
		/*
		while ( ( A_end[a].end == B[i].start ) && (a > 0))
			--a;
		*/

		int num_cant_before = a; 

		while ( ( B_start[b] == end ) && b < B_size)
			++b;

		int num_cant_after = B_size - b;


		int num_left = B_size - num_cant_before - num_cant_after;

		R[i] += num_left;
	}

}
// }}}

//{{{int interval_triple_bsearch_end( struct interval_triple *A_end, 
int interval_triple_bsearch_end( struct interval_triple *A_end, 
								 int A_size,
								 unsigned int key) {
	int lo = -1, hi = A_size, mid;
	while ( hi - lo > 1) {
		mid = (hi + lo) / 2;

		if ( A_end[mid].end < key)
			lo = mid;
		else
			hi = mid;
	}

	return hi;
}
//}}}

//{{{ int interval_triple_bsearch_start( struct interval_triple *A_start, 
int interval_triple_bsearch_start( struct interval_triple *A_start, 
								   int A_size,
								   unsigned int key) {
	int lo = -1, hi = A_size, mid;
	while ( hi - lo > 1) {
		mid = (hi + lo) / 2;

		if ( A_start[mid].start < key )
			lo = mid;
		else
			hi = mid;
	}

	return hi;
}
//}}}

//{{{ int unsigned_bsearch( struct interval_triple *A_start, 
int unsigned_bsearch( unsigned int *A, 
					  int A_size,
					  unsigned int key)
{
	int lo = -1, hi = A_size, mid;
	while ( hi - lo > 1) {
		mid = (hi + lo) / 2;

		if ( A[mid] < key )
			lo = mid;
		else
			hi = mid;
	}

	return hi;
}
//}}}

//{{{ unsigned int map_start_end_from_file( FILE *B_file,
/*
 * This function should read in up to chunk_size lines from B_file, and map
 * them into a 1D space using the universe U by start (in B_start) and end (in
 * B_end).  If there are less then chunk_size lines left in B_file then put
 * the remaining lines into B_start and B_end.  
 *
 * Return B_curr_size
 *
 */
unsigned int map_start_end_from_file_mpi( FILE *B_file,
							 unsigned int *B_start,
							 unsigned int *B_end,
							 unsigned int chunk_size,
							 unsigned int *B_curr_size,
							 struct bed_line *U_array,
							 int U_size,
							 int rank,
							 int net_size,
							 unsigned int *line)
{

	char chr_c[5];
	unsigned int size = 0;
	unsigned int start, end;

	while ( parse_bed_line(B_file, chr_c, &start, &end) &&
			   (size < chunk_size) ) {

		if ( (*line % net_size) == rank ) {

			int chr_i = chr_name_to_int(chr_c);

			int j;
			unsigned int u_start = 0, u_offset = 0;
			int flag = 0;
			for (j = 0; j < U_size; j++) {
				if ( ( U_array[j].chr == chr_i) &&
					 ( U_array[j].start <= end) &&
					 ( U_array[j].end >= start) ) {
					u_start = U_array[j].start;
					u_offset = U_array[j].offset;
					flag = 1;
					break;
				}
			}

			if (flag == 1) {
				B_start[size] = start - u_start + u_offset;
				B_end[size] = end - u_start + u_offset;
				++size;
			}
		}

		*line = *line + 1;
	}

	*B_curr_size = size;

	return size;
}
//}}}

//{{{ int map_start_end_from_file( FILE *B_file,
/*
 * This function should read in up to chunk_size lines from B_file, and map
 * them into a 1D space using the universe U by start (in B_start) and end (in
 * B_end).  If there are less then chunk_size lines left in B_file then put
 * the remaining lines into B_start and B_end.  
 *
 * Return B_curr_size
 *
 */
unsigned int map_start_end_from_file( FILE *B_file,
							 unsigned int *B_start,
							 unsigned int *B_end,
							 unsigned int chunk_size,
							 unsigned int *B_curr_size,
							 struct bed_line *U_array,
							 int U_size)
{

	char chr_c[5];
	unsigned int size = 0;
	unsigned int start, end;

	while ( (size < chunk_size) &&
			parse_bed_line(B_file, chr_c, &start, &end) ) {


		int chr_i = chr_name_to_int(chr_c);

		int j;
		unsigned int u_start = 0, u_offset = 0;
		int flag = 0;
		for (j = 0; j < U_size; j++) {
			if ( ( U_array[j].chr == chr_i) &&
				 ( U_array[j].start <= end) &&
				 ( U_array[j].end >= start) ) {
				u_start = U_array[j].start;
				u_offset = U_array[j].offset;
				flag = 1;
				break;
			}
		}

		if (flag == 1) {
			B_start[size] = start - u_start + u_offset;
			B_end[size] = end - u_start + u_offset;
			++size;
		}
	}


	*B_curr_size = size;

	return size;
}
//}}}
