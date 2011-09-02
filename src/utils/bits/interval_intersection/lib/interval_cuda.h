#ifndef __INTERVAL_CUDA_H__
#define __INTERVAL_CUDA_H__

void per_interval_count_intersections_bsearch_cuda(struct interval *A,
												  unsigned int size_A,
												  struct interval *B,
												  unsigned int size_B,
												  unsigned int *R);

unsigned int count_intersections_bsearch_cuda(struct interval *A,
										      unsigned int size_A,
											  struct interval *B,
										      unsigned int size_B);

unsigned int count_intersections_sort_bsearch_cuda(struct interval *A,
										      unsigned int size_A,
											  struct interval *B,
										      unsigned int size_B);

unsigned int count_intersections_i_gm_bsearch_cuda(struct interval *A,
										      unsigned int size_A,
											  struct interval *B,
										      unsigned int size_B,
										      unsigned int size_I);

void allocate_and_move( struct interval *A,
						unsigned int **A_starts_h,
						unsigned int **A_starts_d,
					   	unsigned int **A_lens_h ,
						unsigned int **A_lens_d,
						unsigned int size_A,
						struct interval *B,
						unsigned int **B_starts_h ,
						unsigned int **B_starts_d,
						unsigned int **B_ends_h ,
						unsigned int **B_ends_d,
						unsigned int size_B,
						unsigned int **R_d);

__global__
void count_bsearch_cuda (	unsigned int *A_start,
							unsigned int *A_len,
							int A_size,
							unsigned int *B_start,
							unsigned int *B_end,
							int B_size,
							unsigned int *R,
							int n);
__global__
void count_i_gm_bsearch_cuda (	unsigned int *A_start,
							unsigned int *A_len,
							int A_size,
							unsigned int *B_start,
							unsigned int *B_end,
							int B_size,
							unsigned int *I_start,
							unsigned int *I_end,
							int I_size,
							unsigned int *R,
							int n);

#endif
