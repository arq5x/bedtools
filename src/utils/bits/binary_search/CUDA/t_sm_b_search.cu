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
			fprintf(stderr, "usage: %s <D size> <Q size> <T size>"
					"<seed> <device>\n",
					argv[0]);
			return 1;
	}

	CUDA_SAFE_CALL( cudaSetDevice( atoi(argv[5] ) ) );
	//CUDA_SAFE_CALL( cudaFree(NULL) );

	int D_size = atoi(argv[1]);
	int Q_size = atoi(argv[2]);
	int T_size = atoi(argv[3]);
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

	//fprintf(stderr, "rand done\n");
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
	//fprintf(stderr, "sort done\n");
	//}}}

	/*
	unsigned int *D_h = (unsigned int *)malloc( D_size * sizeof(unsigned int));
	cudaMemcpy(D_h, D_d, (D_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	*/

	int block_size = 256;
	dim3 dimBlock(block_size);

	int tree_grid_size = ( T_size + block_size - 1) / (block_size * 1);
	dim3 tree_dimGrid( tree_grid_size );

	unsigned int *T_d;
	cudaMalloc((void **)&T_d, (T_size)*sizeof(unsigned int));

	//{{{ tree
	start();
	gen_tree <<<tree_dimGrid, dimBlock>>> ( D_d, D_size, T_d, T_size);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "tree: %s.\n", cudaGetErrorString( err) );
	stop();
	unsigned long tree_time = report();
	//fprintf(stderr, "tree done\n");
	//}}}
	
	/*
	unsigned int *T_h = (unsigned int *) malloc(T_size*sizeof(unsigned int));
	cudaMemcpy(T_h, T_d, (T_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	*/

	int grid_size = ( Q_size + block_size - 1) / (block_size * 1);
	dim3 dimGrid( grid_size );

	unsigned int *R_d;
	cudaMalloc((void **)&R_d, (Q_size)*sizeof(unsigned int));

	//{{{ t_sm_binary_search
	start();
	t_sm_binary_search<<< dimGrid,
						  dimBlock,
						  T_size * sizeof(unsigned int) >>> (
			D_d, D_size, Q_d, Q_size, R_d, T_d, T_size);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "binary_search_gp: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long search_gmtree_1_time = report();
	
	/* DEBUG  START */
	/*
	unsigned int *R_h = (unsigned int *) malloc(Q_size*sizeof(unsigned int));
	cudaMemcpy(R_h, R_d, (Q_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	unsigned int *Q_h = (unsigned int *) malloc(Q_size*sizeof(unsigned int));
	cudaMemcpy(Q_h, Q_d, (Q_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	unsigned int *T_h = (unsigned int *) malloc(T_size*sizeof(unsigned int));
	cudaMemcpy(T_h, T_d, (T_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);

	int x;
	for (x = 0; x < Q_size; x++)
		printf("%d\t%u\t%u\n", x, Q_h[x], R_h[x]);

	for (x = 0; x < T_size; x++)
		printf("%d\t%u\n", x, T_h[x]);
	*/
	/* DEBUG  END */

	start();
	t_gm_binary_search<<< dimGrid,
						  dimBlock,
						  T_size * sizeof(unsigned int) >>> (
			D_d, D_size, Q_d, Q_size, R_d, T_d, T_size);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "binary_search_gp: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long search_gmtree_2_time = report();
	//}}}

	printf("%lu,%lu\n", 
			search_gmtree_1_time + tree_time,
			search_gmtree_2_time + tree_time);

	return 0;
}
