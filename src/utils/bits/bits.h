#ifndef __BITS_H__
#define __BITS_H__

#include <stdint.h>

uint32_t uint32_bsearch(uint32_t key, uint32_t *D, uint32_t D_size, int lo, uint32_t hi);

uint32_t count_intersections_bsearch_seq(uint32_t *A_starts,
                                         uint32_t *A_ends, 
									     uint32_t A_size,
									     uint32_t *B_starts,
									     uint32_t *B_ends,  
                                         uint32_t B_size);

#endif