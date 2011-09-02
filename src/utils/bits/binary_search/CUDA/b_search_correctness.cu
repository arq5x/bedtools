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
			fprintf(stderr, "usage: %s <D size> <Q size> <I/T Size>"
					"<seed> <device>\n",
					argv[0]);
			return 1;
	}

	CUDA_SAFE_CALL( cudaSetDevice( atoi(argv[5] ) ) );
	//CUDA_SAFE_CALL( cudaFree(NULL) );

	int D_size = atoi(argv[1]);
	int Q_size = atoi(argv[2]);
	int I_size = atoi(argv[3]);
	int T_size = I_size;
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
	
	unsigned int *D_h = (unsigned int *)malloc(
			D_size * sizeof(unsigned int));
	cudaMemcpy(D_h, D_d, (D_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	
	unsigned int *Q_h = (unsigned int *)malloc(
			Q_size * sizeof(unsigned int));
	cudaMemcpy(Q_h, Q_d, (Q_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);

	int block_size = 256;
	dim3 dimBlock(block_size);

	int grid_size = ( Q_size + block_size - 1) / (block_size * 1);
	dim3 dimGrid( grid_size );


	//{{{b_search 
	unsigned int *R_d;
	cudaMalloc((void **)&R_d, (Q_size)*sizeof(unsigned int));

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

	unsigned int *BR_h = (unsigned int *)malloc( Q_size * sizeof(unsigned int));
	cudaMemcpy(BR_h, R_d, (Q_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);

	cudaFree(R_d);
	//}}}

	//{{{ index
	int index_grid_size = ( I_size + block_size - 1) / (block_size * 1);
	dim3 index_dimGrid( index_grid_size );

	unsigned int *I_d;
	cudaMalloc((void **)&I_d, (I_size)*sizeof(unsigned int));

	start();
	gen_index <<<index_dimGrid, dimBlock>>> ( D_d, D_size, I_d, I_size);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "index: %s.\n", cudaGetErrorString( err) );
	stop();
	unsigned long index_time = report();

	unsigned int *I_h = (unsigned int *)malloc(
			I_size * sizeof(unsigned int));
	cudaMemcpy(I_h, I_d, (I_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	//}}}

	//{{{ i_gm_binary_search
	unsigned int *IGR_d;
	cudaMalloc((void **)&IGR_d, (Q_size)*sizeof(unsigned int));

	start();
	i_gm_binary_search<<< dimGrid, dimBlock>>> (
			D_d, D_size, Q_d, Q_size, IGR_d, I_d, I_size);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "binary_search_gp 1: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long search_gmindex_1_time = report();

	unsigned int *IGR_h = (unsigned int *)malloc(
			Q_size * sizeof(unsigned int));
	cudaMemcpy(IGR_h, IGR_d, (Q_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	cudaFree(IGR_d);
	//}}}

	//{{{ i_sm_binary_search
	unsigned int *ISR_d;
	cudaMalloc((void **)&ISR_d, (Q_size)*sizeof(unsigned int));

	start();
	i_sm_binary_search<<< dimGrid, dimBlock, I_size * sizeof(unsigned int) >>> (
			D_d, D_size, Q_d, Q_size, ISR_d, I_d, I_size);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "binary_search_gp: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long search_smindex_1_time = report();

	unsigned int *ISR_h = (unsigned int *)malloc(
			Q_size * sizeof(unsigned int));
	cudaMemcpy(ISR_h, ISR_d, (Q_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	cudaFree(ISR_d);

	//}}}

	//{{{ tree
	int tree_grid_size = ( T_size + block_size - 1) / (block_size * 1);
	dim3 tree_dimGrid( tree_grid_size );

	unsigned int *T_d;
	cudaMalloc((void **)&T_d, (T_size)*sizeof(unsigned int));

	start();
	gen_tree <<<tree_dimGrid, dimBlock>>> ( D_d, D_size, T_d, T_size);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "tree: %s.\n", cudaGetErrorString( err) );
	stop();
	unsigned long tree_time = report();

	unsigned int *T_h = (unsigned int *)malloc(
			T_size * sizeof(unsigned int));
	cudaMemcpy(T_h, T_d, (T_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);

	//}}}
	
	//{{{ t_gm_binary_search
	unsigned int *TGR_d;
	cudaMalloc((void **)&TGR_d, (Q_size)*sizeof(unsigned int));

	start();
	t_gm_binary_search<<< dimGrid,
						  dimBlock>>> (
			D_d, D_size, Q_d, Q_size, TGR_d, T_d, T_size);

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

	unsigned int *TGR_h = (unsigned int *)malloc(
			Q_size * sizeof(unsigned int));
	cudaMemcpy(TGR_h, TGR_d, (Q_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	cudaFree(TGR_d);

	//}}}

	//{{{ t_gm_binary_search
	unsigned int *TSR_d;
	cudaMalloc((void **)&TSR_d, (Q_size)*sizeof(unsigned int));

	start();
	t_sm_binary_search<<< dimGrid,
						  dimBlock,
						  T_size * sizeof(unsigned int) >>> (
			D_d, D_size, Q_d, Q_size, TSR_d, T_d, T_size);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "binary_search_gp: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long search_smtree_1_time = report();
	
	unsigned int *TSR_h = (unsigned int *)malloc(
			Q_size * sizeof(unsigned int));
	cudaMemcpy(TSR_h, TSR_d, (Q_size) * sizeof(unsigned int),
			cudaMemcpyDeviceToHost);
	cudaFree(TSR_d);
	//}}}

	int *I = (int *) malloc(I_size * sizeof(int));

	int i;
	for (i = 0; i < Q_size; i++) {
		if (BR_h[i] != IGR_h[i])
				printf(".");
		else
				printf("-");
		if (BR_h[i] != ISR_h[i])
				printf(".");
		else
				printf("-");
		if (BR_h[i] != TGR_h[i])
				printf(".");
		else
				printf("-");
		if (BR_h[i] != TSR_h[i])
				printf(".");
		else
				printf("-");

		printf("\t");
		/*
		if ( (BR_h[i] != IGR_h[i]) ||
			 (BR_h[i] != ISR_h[i]) ||
			 (BR_h[i] != TGR_h[i]) ||
			 (BR_h[i] != TSR_h[i]) )
			printf("-\t");
		else 
			printf("+\t");
		*/

		printf( "%d\t"
				"%u\t"
				"q:%u\t"
				"b:%u\t"
				"ig:%u\t"
				"is:%u\t"
				"tg:%u\t"
				"ts:%u\n",
				i,
				D_h[ BR_h[i] ],
				Q_h[i],
				BR_h[i],
				IGR_h[i],
				ISR_h[i],
				TGR_h[i],
				TSR_h[i]
			  );
		/*
		if ( D_h[ BR_h[i] ] == Q_h[i] )
			printf( "=\t%d\t"
					"d:%u\t"
					"q:%u\t"
					"b:%u\t"
					"ig:%u\t"
					"is:%u\t"
					"tg:%u\t"
					"ts:%u\n",
					i,
					D_h[ BR_h[i] ],
					Q_h[i],
					BR_h[i],
					IGR_h[i],
					ISR_h[i],
					TGR_h[i],
					TSR_h[i]
				  );
			*/
	}

	/*
	for (i = 0; i < I_size - 1; i++) 
			printf("I\t%d\t%d\t%u\n", i, _i_to_I(i,I_size,D_size),I_h[i]);
	*/

	return 0;
}
