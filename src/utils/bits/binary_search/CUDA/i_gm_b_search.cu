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
			fprintf(stderr, "usage: %s <D size> <Q size> <I size>"
					"<seed> <device>\n",
					argv[0]);
			return 1;
	}

	CUDA_SAFE_CALL( cudaSetDevice( atoi(argv[5] ) ) );
	//CUDA_SAFE_CALL( cudaFree(NULL) );

	int D_size = atoi(argv[1]);
	int Q_size = atoi(argv[2]);
	int I_size = atoi(argv[3]);
	int seed = atoi(argv[4]);
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

	/*
	unsigned int *D_h = (unsigned int *)malloc( D_size * sizeof(unsigned int));
	cudaMemcpy(D_h, D_d, (D_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	*/

	int block_size = 256;
	dim3 dimBlock(block_size);

	int index_grid_size = ( I_size + block_size - 1) / (block_size * 1);
	dim3 index_dimGrid( index_grid_size );

	unsigned int *I_d;
	cudaMalloc((void **)&I_d, (I_size)*sizeof(unsigned int));

	//{{{ index
	start();
	gen_index <<<index_dimGrid, dimBlock>>> ( D_d, D_size, I_d, I_size);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "index: %s.\n", cudaGetErrorString( err) );
	stop();
	unsigned long index_time = report();
	//}}}

	/*
	unsigned int *I_h = (unsigned int *) malloc(I_size*sizeof(unsigned int));
	cudaMemcpy(I_h, I_d, (I_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	*/
	

	int grid_size = ( Q_size + block_size - 1) / (block_size * 1);
	dim3 dimGrid( grid_size );

	unsigned int *R_d;
	cudaMalloc((void **)&R_d, (Q_size)*sizeof(unsigned int));

	//{{{ i_gm_binary_search
	start();
	i_gm_binary_search<<< dimGrid, dimBlock>>> (
			D_d, D_size, Q_d, Q_size, R_d, I_d, I_size);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "binary_search_gp 1: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long search_gmindex_1_time = report();

	start();
	i_gm_binary_search<<< dimGrid, dimBlock>>> (
			D_d, D_size, Q_d, Q_size, R_d, I_d, I_size);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "binary_search_gp 2: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long search_gmindex_2_time = report();
	//}}}

	printf("%lu,%lu\n", 
			search_gmindex_1_time + index_time,
			search_gmindex_2_time + index_time);

	return 0;
}
