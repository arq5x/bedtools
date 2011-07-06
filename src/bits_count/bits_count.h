/*****************************************************************************
  bits_count.h

  (c) 2009, 2010, 2011 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0 license.
******************************************************************************/
#ifndef BITS_COUNT_H
#define BITS_COUNT_H

#include "bedFile.h"
#include "bits.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
using namespace std;



class BitsCount {

public:
    // constructor
    BitsCount(string bedAFile, string bedBFile, string genomeFile);
    // destructor
    ~BitsCount(void);

private:

    //------------------------------------------------
    // private attributes
    //------------------------------------------------
    string _bedAFile;
    string _bedBFile;
    string _genomeFile;

    // instance of a bed file class.
    BedFile *_bedA, *_bedB, *_genome;
    
    map<string,CHRPOS> _offsets;

    //------------------------------------------------
    // private methods
    //------------------------------------------------
    void CountOverlaps(void);
};

#endif /* BITS_COUNT_H */
