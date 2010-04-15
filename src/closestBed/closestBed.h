/*****************************************************************************
  closestBed.h

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#ifndef CLOSESTBED_H
#define CLOSESTBED_H

#include "bedFile.h"
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

//************************************************
// Class methods and elements
//************************************************
class BedClosest {

public:

	// constructor 
	BedClosest(string &bedAFile, string &bedAFile, bool &forceStrand, string &tieMode);

	// destructor
	~BedClosest(void);
	
	// find the closest feature in B to A
	void FindClosestBed();
		
private:
	
	// data
	string _bedAFile;
	string _bedBFile;
	string _tieMode;
	bool _forceStrand;
	
	BedFile *_bedA, *_bedB;
	
	// methods
	void reportA(const BED &);
	void reportB(const BED &);
	void reportNullB();
	void FindWindowOverlaps(BED &, vector<BED> &);

};
#endif /* CLOSEST_H */
