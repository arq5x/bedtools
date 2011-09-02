#ifndef __REDUCE_CUDA_H__
#define __REDUCE_CUDA_H__

__global__
void add_unsigned_ints_cuda(unsigned int *gdata,
							unsigned int size,
							unsigned int n );

void parallel_sum( unsigned int *R_d,
				   int block_size,
				   int Rd_size,
				   int n);
#endif
