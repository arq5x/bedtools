#ifndef __BSEARCH_H__
#define __BSEARCH_H__

unsigned int bsearch_seq(unsigned int key,
					  unsigned int *D,
					  unsigned int D_size,
					  int lo,
					  int hi);

unsigned int i_to_I(unsigned int i,
					unsigned int I_size,
					unsigned int D_size);

unsigned int i_to_T(unsigned int i,
					unsigned int T_size,
					unsigned int D_size);

void region_to_hi_lo(unsigned int region,
					 unsigned int I_size,
					 unsigned int D_size,
					 int *D_hi,
					 int *D_lo);

void create_index(unsigned int *D,
				  unsigned int D_size,
				  unsigned int *I,
				  unsigned int I_size);

unsigned int i_bsearch_seq(unsigned int key,
						   unsigned int *D,
						   unsigned int D_size,
						   unsigned int *I,
						   unsigned int I_size);

unsigned int isearch_seq(unsigned int key,
						   unsigned int *I,
						   unsigned int I_size,
						   unsigned int D_size,
						   int *D_hi,
						   int *D_lo);

void create_tree(unsigned int *D,
				 unsigned int D_size,
				 unsigned int *T,
				 unsigned int T_size);

unsigned int tsearch_seq(unsigned int key,
					     unsigned int *T,
					     unsigned int T_size,
					     unsigned int D_size,
					     int *D_hi,
					     int *D_lo,
						 int *hit);

unsigned int t_bsearch_seq(unsigned int key,
					       unsigned int *D,
					       unsigned int D_size,
					       unsigned int *T,
					       unsigned int T_size);

int compare_int(const void *a,
				const void *b);

int compare_unsigned_int(const void *a,
						 const void *b);

void generate_rand_unsigned_int_set(unsigned int *D,
									unsigned int D_size);

#endif
