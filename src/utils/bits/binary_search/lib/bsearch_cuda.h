#ifndef __SET_INTERSECT_CUDA_H__
#define __SET_INTERSECT_CUDA_H__

__device__
int binary_search_cuda( unsigned int *db,
				   int size_db, 
				   unsigned int s);

__global__
void b_search( unsigned int *db,
			   int size_db, 
			   unsigned int *q,
			   int size_q, 
			   unsigned int *R );
/*
__global__
void binary_search_i( unsigned int *db,
					 int size_db, 
					 unsigned int *q,
					 int size_q, 
					 unsigned int *R,
					 int size_I);
*/

__device__
int i_binary_search( unsigned int *db,
					 int size_db, 
					 unsigned int s,
					 unsigned int *I,
					 int size_I);

__device__
int bound_binary_search( unsigned int *db,
				   int size_db, 
				   unsigned int s,
				   int lo,
				   int hi);
__global__
void i_sm_binary_search( unsigned int *db,
					 int size_db, 
					 unsigned int *q,
					 int size_q, 
					 unsigned int *R,
					 unsigned int *I,
					 int size_I);

__global__
void i_gm_binary_search( unsigned int *db,
					 int size_db, 
					 unsigned int *q,
					 int size_q, 
					 unsigned int *R,
					 unsigned int *I,
					 int size_I);

__global__
void t_gm_binary_search( unsigned int *db,
					 int size_db, 
					 unsigned int *q,
					 int size_q, 
					 unsigned int *R,
					 unsigned int *T,
					 int size_T);

__global__
void gen_index( unsigned int *db,
			    int size_db, 
				unsigned int *I,
				int size_I);
				     
__device__
unsigned int i_to_T(int i,
					int T_size,
					int D_size);
 
__device__
unsigned int i_to_I(int i,
					int I_size,
					int D_size);

__global__
void gen_tree(unsigned int *db,
			  int size_db, 
			  unsigned int *T,
			  int size_T);

__device__
void region_to_hi_lo(int region,
					 int I_size,
					 int D_size,
					 int *D_hi,
					 int *D_lo);
__global__
void t_gm_binary_search( unsigned int *db,
					 int size_db, 
					 unsigned int *q,
					 int size_q, 
					 unsigned int *R,
					 unsigned int *T,
					 int size_T);
__global__
void t_sm_binary_search( unsigned int *db,
					 int size_db, 
					 unsigned int *q,
					 int size_q, 
					 unsigned int *R,
					 unsigned int *T,
					 int size_T);
					     
unsigned int _i_to_T(int i,
					int T_size,
					int D_size);
 
unsigned int _i_to_I(int i,
					int I_size,
					int D_size);
#endif
