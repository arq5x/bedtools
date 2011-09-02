#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil.h>
#include <sys/time.h>
#include "radixsort.h"
#include "random.hpp"
#include "timer.h"
#include "bsearch_cuda.h"

int main(int argc, char *argv[]) {


	if (argc < 4) {
			fprintf(stderr, "usage: %s <D size> <Q size> "
					"<seed> <device>\n",
					argv[0]);
			return 1;
	}

	CUDA_SAFE_CALL( cudaSetDevice( atoi(argv[4] ) ) );
	//CUDA_SAFE_CALL( cudaFree(NULL) );

	int D_size = atoi(argv[1]);
	int Q_size = atoi(argv[2]);
	int seed = atoi(argv[3]);
	cudaError_t err;

	//{{{ gen Q and D
	RNG_rand48 D_r(seed);
	D_r.generate(D_size);
	unsigned int *D_d = (unsigned int *)D_r.get_random_numbers();

	RNG_rand48 Q_r(seed);
	Q_r.generate(Q_size);
	unsigned int *Q_d = (unsigned int *)Q_r.get_random_numbers();
	
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "rand errors: %s.\n", cudaGetErrorString( err) );
	//}}}

	//{{{ sort Q
	start();
	nvRadixSort::RadixSort sort_Q_d(Q_size, true);
	sort_Q_d.sort((unsigned int*)Q_d, 0, Q_size, 32);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "sort q: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long sort_q_time = report();
	//}}}
	
	//{{{ sort D
	start();
	nvRadixSort::RadixSort sort_D_d(D_size, true);
	sort_D_d.sort((unsigned int*)D_d, 0, D_size, 32);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "sort d: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long sort_d_time = report();
	//}}}

	unsigned int *D_h = (unsigned int *)malloc( D_size * sizeof(unsigned int));
	cudaMemcpy(D_h, D_d, (D_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);

	int block_size = 256;
	dim3 dimBlock(block_size);

	int grid_size = ( Q_size + block_size - 1) / (block_size * 1);
	dim3 dimGrid( grid_size );

	unsigned int *R_d;
	cudaMalloc((void **)&R_d, (Q_size)*sizeof(unsigned int));

	//{{{b_search 
	start();
	b_search <<<dimGrid, dimBlock>>> (D_d, D_size, Q_d, Q_size, R_d);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "b_search: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long search_noindex_1_time = report();

	start();
	b_search <<<dimGrid, dimBlock>>> (D_d, D_size, Q_d, Q_size, R_d);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "b_search: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long search_noindex_2_time = report();
	//}}}

	printf("%lu,%lu\n", 
			search_noindex_1_time + sort_q_time,
			search_noindex_2_time + sort_q_time);

	return 0;
}
