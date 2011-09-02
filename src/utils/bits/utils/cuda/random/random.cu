/*
  Copyright (c) 2007 A. Arnold and J. A. van Meel, FOM institute
  AMOLF, Amsterdam; all rights reserved unless otherwise stated.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  In addition to the regulations of the GNU General Public License,
  publications and communications based in parts on this program or on
  parts of this program are required to cite the article
  "Harvesting graphics power for MD simulations"
  by J.A. van Meel, A. Arnold, D. Frenkel, S. F. Portegies Zwart and
  R. G. Belleman, Molecular Simulation, Vol. 34, p. 259 (2007).

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
  MA 02111-1307 USA
*/
#include "gpu.hpp"
#include "random.hpp"

/************************************************
 * GPU kernels
 ************************************************/

/************************************************
 * RNG iteration
 ************************************************/

/** propagate an rand48 RNG one iteration.
    @param Xn  the current RNG state, in 2x 24-bit form
    @param A,C the magic constants for the RNG. For striding,
               this constants have to be adapted, see the constructor
    @result    the new RNG state X(n+1)
*/
__device__
static uint2 RNG_rand48_iterate_single(uint2 Xn, uint2 A, uint2 C)
{
  // results and Xn are 2x 24bit to handle overflows optimally, i.e.
  // in one operation.

  // the multiplication commands however give the low and hi 32 bit,
  // which have to be converted as follows:
  // 48bit in bytes = ABCD EF (space marks 32bit boundary)
  // R0             = ABC
  // R1             =    D EF

  unsigned int R0, R1;

  // low 24-bit multiplication
  const unsigned int lo00 = __umul24(Xn.x, A.x);
  const unsigned int hi00 = __umulhi(Xn.x, A.x);

  // 24bit distribution of 32bit multiplication results
  R0 = (lo00 & 0xFFFFFF);
  R1 = (lo00 >> 24) | (hi00 << 8);

  R0 += C.x; R1 += C.y;

  // transfer overflows
  R1 += (R0 >> 24);
  R0 &= 0xFFFFFF;

  // cross-terms, low/hi 24-bit multiplication
  R1 += __umul24(Xn.y, A.x);
  R1 += __umul24(Xn.x, A.y);

  R1 &= 0xFFFFFF;

  return make_uint2(R0, R1);
}

/************************************************
 * sets of random numbers
 ************************************************/

/** create a set of random numbers. The random numbers are generated in blocks.
    In each block, a thread calculates one random number, the first thread the
    first one, the second the second and so on.
    @param state      the current states of the RNGS, one per thread.
    @param res        where to put the generated numbers
    @param num_blocks how many random numbers each thread generates.
                      The total number of random numbers is therefore
		      num_blocks*nThreads.
    @param A,C        the magic constants for the iteration. They need
                      to be chosen as to advance the RNG by nThreads iterations
		      at once, see the constructor.
*/
__global__
static void RNG_rand48_get_int(uint2 *state, int *res, int num_blocks, uint2 A, uint2 C)
{
  const int nThreads = blockDim.x*gridDim.x;

  // load the current state of the RNG into a register
  int   nOutIdx = threadIdx.x + blockIdx.x*blockDim.x;
  uint2 lstate = state[nOutIdx];
  int i;
  for (i = 0; i < num_blocks; ++i) {
    // get upper 31 (!) bits of the 2x 24bits
    res[nOutIdx] = ( lstate.x >> 17 ) | ( lstate.y << 7);
    nOutIdx += nThreads;
    // this actually iterates the RNG
    lstate = RNG_rand48_iterate_single(lstate, A, C);
  }

  nOutIdx = threadIdx.x + blockIdx.x*blockDim.x;
  state[nOutIdx] = lstate;
}

/************************************************
 * RNG_rand48 implementation
 ************************************************/

void
RNG_rand48::init(int seed)
{
  // setup execution grid to get max performance
  threadsX = 192;
  blocksX  = 32;

  const int nThreads = threadsX*blocksX;

  uint2* seeds = new uint2[ nThreads ];

  CUDA_SAFE_CALL( cudaMalloc( (void**) &state, sizeof(uint2)*nThreads ) );

  // calculate strided iteration constants
  unsigned long long A, C;
  A = 1LL; C = 0LL;
  for (unsigned int i = 0; i < nThreads; ++i) {
    C += A*c;
    A *= a;
  }
  A0 = A & 0xFFFFFFLL;
  A1 = (A >> 24) & 0xFFFFFFLL;
  C0 = C & 0xFFFFFFLL;
  C1 = (C >> 24) & 0xFFFFFFLL;

  // prepare first nThreads random numbers from seed
  unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
  for (unsigned int i = 0; i < nThreads; ++i) {
    x = a*x + c;
    seeds[i].x = x & 0xFFFFFFLL;
    seeds[i].y = (x >> 24) & 0xFFFFFFLL;
  }

  CUDA_SAFE_CALL(cudaMemcpy(state, seeds, sizeof(uint2)*nThreads, cudaMemcpyHostToDevice));

  delete[] seeds;
}

void
RNG_rand48::cleanup() {
  CUDA_SAFE_CALL(cudaFree((void*) state));
}

void
RNG_rand48::generate(int n)
{
  const int nThreads = threadsX*blocksX;

  int num_blocks = (n + nThreads-1)/nThreads;
	
  if (res == 0) {
    CUDA_SAFE_CALL(cudaMalloc( (void**) &res, sizeof(int)*nThreads*num_blocks));
  }
  
  dim3 grid( blocksX, 1, 1);
  dim3 threads( threadsX, 1, 1);

  uint2 A, C;
  A.x = A0; A.y = A1;
  C.x = C0; C.y = C1;

  // call GPU kernel
  RNG_rand48_get_int<<< grid, threads >>>((uint2 *)state, (int *)res, num_blocks, A, C);
}

void
RNG_rand48::get(int *r, int n)
 {
  CUDA_SAFE_CALL(cudaMemcpy( r, res, sizeof(int)*n, cudaMemcpyDeviceToHost ) );
}
