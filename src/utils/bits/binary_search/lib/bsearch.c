#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bsearch.h"
#include "mt.h"

//{{{ unsigned int bsearch_seq(unsigned int key,
unsigned int bsearch_seq(unsigned int key,
						 unsigned int *D,
						 unsigned int D_size,
						 int lo,
						 int hi)
{
	//int lo = -1, hi = D_size, mid;
	int i = 0;
	unsigned int mid;
	//fprintf(stderr, "k:%d\t", key);
	while ( hi - lo > 1) {
		++i;
		mid = (hi + lo) / 2;
		//printf("%d\t", D[mid]);
		//fprintf(stderr, "%d,%d,%d\t", lo, mid, hi, D[mid]);
		if ( D[mid] < key )
			lo = mid;
		else
			hi = mid;
	}
	//fprintf(stderr, "\n");
	
	return hi;
}
//}}}

//{{{ unsigned int i_to_I(unsigned int i,
unsigned int i_to_I(unsigned int i,
					unsigned int I_size,
					unsigned int D_size)
{
	//int lo = -1, hi = D_size;
	//unsigned int regions = I_size + 1;
	//unsigned int hi = D_size;
	//unsigned int r =( (i+1)*hi - (regions - (i+1))) / (regions);
	//return r;


	unsigned long int regions = I_size + 1;
	unsigned long int hi = D_size;
	unsigned long int j = i;
	unsigned int r =( (j+1)*hi - (regions - (j+1))) / (regions);
	return r;
}
//}}}

//{{{ unsigned int i_to_T(unsigned int i,
unsigned int i_to_T(unsigned int i,
					unsigned int T_size,
					unsigned int D_size)
{
	//int lo = -1, hi = D_size;
	//double row_d = log(i + 1) / log(2);
	//unsigned int row = (int) (row_d);
	//unsigned int prev = pow(2, row) - 1;
	//unsigned int i_row = i - prev;

	//unsigned int hi_v = 2*i_row + 1;
	//unsigned int lo_v = pow(2, row + 1) - (2*i_row +1);
	//unsigned int div = pow(2,row + 1);
	//unsigned int r = ( hi_v*hi - lo_v) / div;

	//printf("hi:%d\tlo:%d\trow:%u\thi_v:%u\tlo_v:%u\tdiv:%u\tr:%u\n",
			//hi, lo, row, hi_v, lo_v, div, r);

	//return r;

	unsigned long int hi = D_size;
	double row_d = log(i + 1) / log(2);
	unsigned long int row = (unsigned long int) (row_d);
	unsigned long int prev = pow(2, row) - 1;
	unsigned long int i_row = i - prev;

	unsigned long int hi_v = 2*i_row + 1;
	unsigned long int lo_v = pow(2, row + 1) - (2*i_row +1);
	unsigned long int div = pow(2,row + 1);
	unsigned long int r = ( hi_v*hi - lo_v) / div;

	//printf("hi:%d\tlo:%d\trow:%u\thi_v:%u\tlo_v:%u\tdiv:%u\tr:%u\n",
			//hi, lo, row, hi_v, lo_v, div, r);
	return r; 

}
//}}}

//{{{ void region_to_hi_lo(unsigned int region,
void region_to_hi_lo(unsigned int region,
					 unsigned int I_size,
					 unsigned int D_size,
					 int *D_hi,
					 int *D_lo)
{
	unsigned long int hi = D_size;
	unsigned long int r = region;

	unsigned long int l_new_hi = ((r+1)*hi - (I_size - (r+1)) ) / I_size;
	//int new_hi = ( (region+1)*hi - (I_size - (region+1)) ) / I_size;
	//int new_lo = (( (region)*hi - (I_size - (region+1)) ) / I_size) - 1;
	unsigned long int l_new_lo = (((r)*hi - (I_size - (r+1)) ) / I_size) - 1;

	int new_hi = l_new_hi, new_lo = l_new_lo;

	if (region == 0) {
		new_hi = l_new_hi;
		new_lo = -1;
	} else if (region == I_size) {
		new_hi = D_size;
		l_new_lo = ( (r-1)*hi - (I_size - (r+1)) ) / I_size;
		new_lo = l_new_lo;
	}
	
	*D_hi = new_hi;
	*D_lo = new_lo;

	/*
	unsigned int hi = D_size;
	//int new_hi = ( (region+1)*hi + (I_size - (region+1))*lo ) / I_size;
	//int new_lo = (( (region)*hi + (I_size - (region+1))*lo ) / I_size) - 1;
	int new_hi = ( (region+1)*hi - (I_size - (region+1)) ) / I_size;
	int new_lo = (( (region)*hi - (I_size - (region+1)) ) / I_size) - 1;

	//--new_lo;

	if (region == 0)
		new_lo = -1;
	else if (region == I_size) {
		new_hi = D_size;
		//new_lo = ( (region-1)*hi + (I_size - (region+1))*lo ) / I_size;
		new_lo = ( (region-1)*hi - (I_size - (region+1)) ) / I_size;
		//--new_lo;
	}
	

	*D_hi = new_hi;
	*D_lo = new_lo;
	*/
}
//}}}

//{{{ void create_index(unsigned int *D,
void create_index(unsigned int *D,
				  unsigned int D_size,
				  unsigned int *I,
				  unsigned int I_size)
{
	int j;
	for (j = 0; j < I_size; j++) 
		I[ j ] = D[ i_to_I(j,I_size,D_size) ];
}
//}}}

//{{{ unsigned int i_bsearch_seq(unsigned int key,
unsigned int i_bsearch_seq(unsigned int key,
						   unsigned int *D,
						   unsigned int D_size,
						   unsigned int *I,
						   unsigned int I_size)
{
	int hi, lo;
	//unsigned int region = isearch_seq(key, I, I_size + 1, D_size, &hi, &lo);
	unsigned int region = isearch_seq(key, I, I_size, D_size, &hi, &lo);
	//printf("i:%d,%d,%d\n",region,lo,hi);
	return bsearch_seq(key, D, D_size, lo, hi);
}
//}}}

//{{{ unsigned int isearch_seq(unsigned int key,
unsigned int isearch_seq(unsigned int key,
						 unsigned int *I,
						 unsigned int I_size,
						 unsigned int D_size,
						 int *D_hi,
						 int *D_lo)
{
	unsigned int region = bsearch_seq(key, I, I_size, -1, I_size);
	//printf("i\tr:%u\tI_size:%u\n",region, I_size);
	region_to_hi_lo(region, I_size + 1, D_size, D_hi, D_lo);
	return region;
}
//}}}

//{{{ void create_tree(unsigned int *D,
void create_tree(unsigned int *D,
				 unsigned int D_size,
				 unsigned int *T,
				 unsigned int T_size)
{
	int j;
	for (j = 0; j < T_size; j++) 
		T[ j ] = D[ i_to_T(j, T_size, D_size) ];
}
//}}}

//{{{ unsigned int tsearch_seq(unsigned int key,
unsigned int tsearch_seq(unsigned int key,
					     unsigned int *T,
					     unsigned int T_size,
					     unsigned int D_size,
					     int *D_hi,
					     int *D_lo,
						 int *hit)
{
	unsigned long int b = 0;
	unsigned long int size = T_size;

	//printf("t:");
	while (b < size) {
		//printf("%d,%d\t", T[b],b);
		if (key < T[b])
			b = 2*b + 1;
		else if (key > T[b])
			b = 2*b + 2;
		else
			break;
	}

	unsigned int region = b - size;
	
	//printf("%d\t%d\n", b, region);
	//region_to_hi_lo(b - T_size, T_size + 1, D_size, D_hi, D_lo);

	//printf("t\tr:%u\tT_size:%u\n",region, T_size);
	if (T[b] == key) {
		*hit = 1;
		//fprintf(stderr, "HIT\tk:%u\tb:%lu\tT:%u\n",key,b,T[b]);
		return b;
	} else {
		//fprintf(stderr, "NO HIT\n");
		*hit = 0;
		region_to_hi_lo(region, T_size + 1, D_size, D_hi, D_lo);
		return region;
	}
}
//}}}

//{{{ unsigned int t_bsearch_seq(unsigned int key,
unsigned int t_bsearch_seq(unsigned int key,
					       unsigned int *D,
					       unsigned int D_size,
					       unsigned int *T,
					       unsigned int T_size)
{
	int hi, lo, hit;
	unsigned int region = tsearch_seq(key, T, T_size, D_size, &hi, &lo, &hit);
	//printf("t:%d,%d,%d\n",region,lo,hi);
	if (hit == 1)
		return i_to_T(region, T_size, D_size);
	else
		return bsearch_seq(key, D, D_size, lo, hi);
}
//}}}

//{{{int compare_int(const void *a, const void *b)
int compare_int(const void *a, const void *b)
{
	int *a_i = (int *)a;
	int *b_i = (int *)b;

	return *a_i - *b_i;
}
//}}}

//{{{int compare_unsigned_int (const void *a, const void *b)
int compare_unsigned_int (const void *a, const void *b)
{  
	unsigned int *a_i = (unsigned int *)a;
	unsigned int *b_i = (unsigned int *)b;
	if (*a_i < *b_i)
		return -1;
	else if (*a_i > *b_i)
		return 1;
	else
		return 0;
}
//}}}

//{{{ void generate_rand_unsigned_int_set(unsigned int *D,
void generate_rand_unsigned_int_set(unsigned int *D,
									 unsigned int D_size)
{
	int j;
	for (j = 0; j < D_size; j++)
		D[j] = genrand_int32();
}
//}}}
