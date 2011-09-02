#include <stdio.h>
#include <cuda.h>
#include "reduce.h"

//{{{void parallel_sum( int *R_d,
/**
 * @param R_d Address of element array on device
 * @param block_size Number of threads per block
 * @param Rd_size Number of elemens in R_d
 * @param n Number of elemens each thread handles
 */
void parallel_sum( unsigned int *R_d,
				   int block_size,
				   int Rd_size,
				   int n)
{
	unsigned int left = Rd_size;
	while (left > 1) {

		int grid_size = ( left + block_size*n - 1) / (block_size * n);
		dim3 dimGridR( grid_size);

		dim3 dimBlockR( block_size );
		size_t sm_size = dimBlockR.x * sizeof(int); 

		add_unsigned_ints_cuda <<<dimGridR, dimBlockR, sm_size>>>
			(R_d, left, n);

		cudaThreadSynchronize();
		cudaError_t err;
		err = cudaGetLastError();
		if(err != cudaSuccess)
			fprintf(stderr, "My Reduce: %s.\n", cudaGetErrorString( err) );

		left = dimGridR.x;
	}
}
//}}}

//{{{__global__ void my_reduce( int *gdata,
__global__
void add_unsigned_ints_cuda(unsigned int *gdata,
							unsigned int size,
							unsigned int n )
{
	extern __shared__ int sdata[];


	/* v1 load:  need N threads
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x *  blockDim.x  + tid;

	if (i < size)
		sdata[tid] = gdata[i];
	else
		sdata[tid] = 0;
	__syncthreads();
	*/

	/* v2 load:  need N/2 threads 
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
	if (i < size)
		sdata[tid] = gdata[i];
	else
		sdata[tid] = 0;

	if (i + blockDim.x < size)
		sdata[tid] += gdata[i + blockDim.x];
	else
		sdata[tid] += 0;

	__syncthreads();
	*/

	/* v3 load: need N/n threads */
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * ( 2 * blockDim.x ) + tid;
	unsigned int grid_size = blockDim.x * ( 2 * gridDim.x);

	sdata[tid] = 0;

	while ( i < (n * grid_size) ) {
		if (i < size)
			sdata[tid] += gdata[i];

		if ( (i + blockDim.x) < size)
			sdata[tid] += gdata[i + blockDim.x];
		i += grid_size;
	}
	__syncthreads();


	/* v1 calc
	unsigned int s;
	for (s = 1; s < blockDim.x; s*=2) {
		if (tid % (2*s) == 0)
			sdata[tid] += sdata[tid + s];

		__syncthreads();
	}
	*/
	/* v2 calc
	unsigned int s;
	for (s = 1; s < blockDim.x; s*=2) {
		int index = 2 * s * tid;
		if (index < blockDim.x)
			sdata[index] += sdata[index + s];

		__syncthreads();
	}
	*/

	/* v3 calc */
	unsigned int s;
	for (s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];

		__syncthreads();
	}


	/* v5 calc
	if (blockDim.x >= 512) {
		if (tid < 256)
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if (blockDim.x >= 128) {
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}

	if (tid < 32) {
		if (blockDim.x >= 64)
			sdata[tid] += sdata[tid + 32];
		if (blockDim.x >= 32)
			sdata[tid] += sdata[tid + 16];
		if (blockDim.x >= 16)
			sdata[tid] += sdata[tid + 8];
		if (blockDim.x >= 8)
			sdata[tid] += sdata[tid + 4];
		if (blockDim.x >= 4)
			sdata[tid] += sdata[tid + 2];
		if (blockDim.x >= 4)
			sdata[tid] += sdata[tid + 2];
		if (blockDim.x >= 2)
			sdata[tid] += sdata[tid + 1];
	} */

	if (tid == 0)
		gdata[blockIdx.x] = sdata[0];
}
//}}}
