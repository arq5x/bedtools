#ifndef __INTERVAL_H__
#define __INTERVAL_H__

struct interval {
	unsigned int start, end;
};

int compare_interval_by_start (const void *a, const void *b);

unsigned int per_interval_count_intersections_bsearch_seq(struct interval *A,
														   unsigned int size_A,
														   struct interval *B,
														   unsigned int size_B,
														   unsigned int *R);
void enumerate_intersections_bsearch_seq(struct interval *A,
										 unsigned int size_A,
										 struct interval *B,
										 unsigned int size_B,
										 unsigned int *R,
										 unsigned int *E,
										 unsigned int size_E);

unsigned int count_intersections_bsearch_seq(struct interval *A,
										     unsigned int size_A,
											 struct interval *B,
										     unsigned int size_B);

unsigned int count_intersections_i_bsearch_seq(struct interval *A,
										       unsigned int size_A,
											   struct interval *B,
										       unsigned int size_B,
											   unsigned int size_I);

unsigned int count_intersections_t_bsearch_seq(struct interval *A,
										       unsigned int size_A,
											   struct interval *B,
										       unsigned int size_B,
											   unsigned int size_T);

unsigned int count_intersections_sort_bsearch_seq(struct interval *A,
										          unsigned int size_A,
											      struct interval *B,
										          unsigned int size_B);

unsigned int count_intersections_brute_force_seq(struct interval *A,
												 unsigned int size_A,
											     struct interval *B,
										         unsigned int size_B);
void generate_interval_sets(struct interval *A,
							unsigned int size_A,
							unsigned int len_A,
							struct interval *B,
							unsigned int size_B,
							unsigned int len_B,
							unsigned int P);

void generate_ind_interval_sets(struct interval *A,
							unsigned int size_A,
							unsigned int len_A,
							struct interval *B,
							unsigned int size_B,
							unsigned int len_B);

#endif
