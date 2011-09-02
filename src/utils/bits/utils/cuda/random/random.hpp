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
#ifndef RANDOM_HPP
#define RANDOM_HPP

/** a rand48 random number generator.
    The random number generator works similar to the standard lrand48.
    The seed, which is normally set by srand48, here a parameter of the constructor.
    Random numbers are drawn in two steps:
    - first, random numbers are generated via the generate function, and then
    - then, they can be retrieved via the get function
    Alternatively, you can use them directly on the GPU. A pointer to the random
    number array can be retrieved by get_random_numbers. This functions returns a
    void *, to avoid CUDA data types in this header file. The true type
    is however int *.
*/
class RNG_rand48 {
  int stride;

  /// current random numbers per thread
  void *state;
  /// generated random numbers
  void *res;

  /// number of threads
  int  threadsX;
  /// number of blocks of threads
  int  blocksX;

  /** strided iteration constants
      (48-bit, distributed on 2x 24-bit) */
  unsigned int A0, A1, C0, C1;

  /// magic constants for rand48
  static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;

  /// CUDA-safe constructor
  void init(int seed);
  /// CUDA-safe destructor
  void cleanup();

public:
  /// initialize the RNG with seed, just as the standard srand48-function
  RNG_rand48(int seed): res(0) { init(seed); }
  ///
  ~RNG_rand48() { cleanup(); }

  /// generate n random numbers as 31-bit integers like lrand48
  void generate(int n);
  /** get the first n of the previously generated numbers into array r.
      r must be large enough to contain all the numbers, and enough
      numbers have to be generated before. */
  void get(int *r, int n);

  /** return a GPU pointer to the generated random numbers, for
      using them in other GPU functions. */
  void *get_random_numbers() { return res; }
};

#endif
