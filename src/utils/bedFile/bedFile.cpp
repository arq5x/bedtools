// 
//  bedFile.cpp
//  BEDTools
//  
//  Created by Aaron Quinlan Spring 2009.
//  Copyright 2009 Aaron Quinlan. All rights reserved.
//
//  Summary:  Contains common functions for finding BED overlaps.
//
//  Acknowledgments: Much of the code herein is taken from Jim Kent's
//                   BED processing code.  I am grateful for his elegant
//					 genome binning algorithm and therefore use it extensively.

#include "lineFileUtilities.h"
#include "bedFile.h"

static int binOffsetsExtended[] =
	{4096+512+64+8+1, 512+64+8+1, 64+8+1, 8+1, 1, 0};

	
#define _binFirstShift 17	/* How much to shift to get to finest bin. */
#define _binNextShift 3		/* How much to shift to get to next larger bin. */


// return the amount of overlap between two features.  Negative if none.
int overlaps(const int aS, const int aE, const int bS, const int bE) {
	return min(aE, bE) - max(aS, bS);
}


bool leftOf(const int a, const int b) {
	return (a < b);
}

// return the lesser of two values.
int min(const int a, int b) {
	if (a <= b) {
		return a;
	}
	else {
		return b;
	}
}

// return the greater of two values.
int max(const int a, int b) {
	if (a >= b) {
		return a;
	}
	else {
		return b;
	}
}


static int getBin(int start, int end)
/* 
	NOTE: Taken ~verbatim from kent source.
	
	Given start,end in chromosome coordinates assign it
	* a bin.   There's a bin for each 128k segment, for each
	* 1M segment, for each 8M segment, for each 64M segment,
	* and for each chromosome (which is assumed to be less than
	* 512M.)  A range goes into the smallest bin it will fit in. */
{
	int startBin = start, endBin = end-1, i;
	startBin >>= _binFirstShift;
	endBin >>= _binFirstShift;
	
	for (i=0; i<6; ++i) {
		if (startBin == endBin) {
			return binOffsetsExtended[i] + startBin;
		}
		startBin >>= _binNextShift;
		endBin >>= _binNextShift;
	}
	
	cerr << "start " << start << ", end " << end << " out of range in findBin (max is 512M)" << endl;
	return 0;
}

//*********************************************
// Sorting functions
//*********************************************
bool sortByChrom(BED const & a, BED const & b){
	if (a.chrom < b.chrom) return true;
	else return false;
};

bool sortByStart(const BED &a, const BED &b){
	if (a.start < b.start) return true;
	else return false;
};

bool sortBySizeAsc(const BED &a, const BED &b){
	
	unsigned int aLen = a.end - a.start;
	unsigned int bLen = b.end - b.start;
	
	if (aLen < bLen) return true;
	else return false;
};

bool sortBySizeDesc(const BED &a, const BED &b){
	
	unsigned int aLen = a.end - a.start;
	unsigned int bLen = b.end - b.start;
	
	if (aLen > bLen) return true;
	else return false;
};

bool sortByScoreAsc(const BED &a, const BED &b){
	if (a.score < b.score) return true;
	else return false;
};

bool sortByScoreDesc(const BED &a, const BED &b){
	if (a.score > b.score) return true;
	else return false;
};


bool byChromThenStart(BED const & a, BED const & b){

	if (a.chrom < b.chrom) return true;
	else if (a.chrom > b.chrom) return false;

	if (a.start < b.start) return true;
	else if (a.start >= b.start) return false;

	return false;
};


/*
	NOTE: Taken ~verbatim from kent source.
	Return a list of all items in binKeeper that intersect range.
	
	Free this list with slFreeList.
*/
void BedFile::binKeeperFind(map<int, vector<BED>, std::less<int> > &bk, const int start, const int end, vector<BED> &hits) {

	int startBin, endBin;
	int i,j;

	startBin = (start>>_binFirstShift);
	endBin = ((end-1)>>_binFirstShift);
	for (i=0; i<6; ++i) {
		int offset = binOffsetsExtended[i];

		for (j = (startBin+offset); j <= (endBin+offset); ++j)  {
			for (vector<BED>::iterator el = bk[j].begin(); el != bk[j].end(); ++el) {
				if (overlaps(el->start, el->end, start, end) > 0) {
					hits.push_back(*el);
				}
			}
		}
		startBin >>= _binNextShift;
		endBin >>= _binNextShift;
	}
}


void BedFile::countHits(map<int, vector<BED>, std::less<int> > &bk, const int start, const int end) {
	int startBin, endBin;
	int i,j;

	startBin = (start>>_binFirstShift);
	endBin = ((end-1)>>_binFirstShift);
	for (i=0; i<6; ++i) {
		int offset = binOffsetsExtended[i];

		for (j = (startBin+offset); j <= (endBin+offset); ++j) {
			for (vector<BED>::iterator el = bk[j].begin(); el != bk[j].end(); ++el) {
				if (overlaps(el->start, el->end, start, end) > 0) {
					el->count++;
				}
			}
		}
		startBin >>= _binNextShift;
		endBin >>= _binNextShift;
	}
}


// Constructor
BedFile::BedFile(string &bedFile) {
	this->bedFile = bedFile;
}

// Destructor
BedFile::~BedFile(void) {
}


bool BedFile::parseBedLine (BED &bed, const vector<string> &lineVector, const int &lineNum) {

	if ( (lineNum > 1) && (lineVector.size() == this->bedType)) {

		if (this->bedType == 3) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = "";
			bed.score = 0;
			bed.strand = "+";
			return true;
		}
		else if (this->bedType == 4) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = 0;
			bed.strand = "+";
			return true;
		}
		else if (this->bedType ==5) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = atoi(lineVector[4].c_str());
			bed.strand = "+";
			return true;			
		}
		else if (this->bedType == 6) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = atoi(lineVector[4].c_str());
			bed.strand = lineVector[5];
			return true;
		}
		else {
			cerr << "Unexpected number of fields: " << lineNum << ".  Verify that your files are TAB-delimited and that your BED file has 3,4,5 or 6 fields.  Exiting..." << endl;
			exit(1);
		}
		
		if (bed.start > bed.end) {
			cerr << "Error: malformed BED entry at line " << lineNum << ". Start was greater than End. Ignoring it and moving on." << endl;
			return false;
		}
		else if ( (bed.start < 0) || (bed.end < 0) ) {
			cerr << "Error: malformed BED entry at line " << lineNum << ". Coordinate <= 0. Ignoring it and moving on." << endl;
			return false;
		}
	}
	else if ((lineNum == 1) && (lineVector.size() >= 3)) {
		this->bedType = lineVector.size();

		if (this->bedType == 3) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			return true;
		}
		else if (this->bedType == 4) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			return true;
		}
		else if (this->bedType ==5) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = atoi(lineVector[4].c_str());
			return true;			
		}
		else if (this->bedType == 6) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = atoi(lineVector[4].c_str());
			bed.strand = lineVector[5];
			return true;
		}
		else {
			cerr << "Unexpected number of fields: " << lineNum << ".  Verify that your files are TAB-delimited and that your BED file has 3,4,5 or 6 fields.  Exiting..." << endl;
			exit(1);
		}
		
		if (bed.start > bed.end) {
			cerr << "Error: malformed BED entry at line " << lineNum << ". Start was greater than End. Ignoring it and moving on." << endl;
			return false;
		}
		else if ( (bed.start < 0) || (bed.end < 0) ) {
			cerr << "Error: malformed BED entry at line " << lineNum << ". Coordinate <= 0. Ignoring it and moving on." << endl;
			return false;
		}
	}
	else if (lineVector.size() == 1) {
		cerr << "Only one BED field detected: " << lineNum << ".  Verify that your files are TAB-delimited.  Exiting..." << endl;
		exit(1);		
	}
	else if ((lineVector.size() != this->bedType) && (lineVector.size() != 0)) {
		cerr << "Differing number of BED fields encountered at line: " << lineNum << ".  Exiting..." << endl;
		exit(1);
	}
	else if ((lineVector.size() < 3) && (lineVector.size() != 0)) {
		cerr << "TAB delimited BED file with at least 3 fields (chrom, start, end) is required at line: "<< lineNum << ".  Exiting..." << endl;
		exit(1);
	}
	return false;
}


void BedFile::loadBedFileIntoMap() {

	// open the BED file for reading                                                                                                                                      
	ifstream bed(bedFile.c_str(), ios::in);
	if ( !bed ) {
		cerr << "Error: The requested bed file (" <<bedFile << ") could not be opened. Exiting!" << endl;
		exit (1);
	}

	string bedLine;
	BED bedEntry;                                                                                                                        
	int lineNum = 0;

	vector<string> bedFields;	// vector of strings for each column in BED file.
	bedFields.reserve(6);		// reserve space for worst case (BED 6)

	while (getline(bed, bedLine)) {

		if ((bedLine.find_first_of("track") == 1) || (bedLine.find_first_of("browser") == 1)) {
			continue;
		}
		else {
			Tokenize(bedLine,bedFields);
			lineNum++;

			if (parseBedLine(bedEntry, bedFields, lineNum)) {
				int bin = getBin(bedEntry.start, bedEntry.end);
				bedEntry.count = 0;
				this->bedMap[bedEntry.chrom][bin].push_back(bedEntry);	
			}
			bedFields.clear();
		}
	}
}


void BedFile::loadBedFileIntoMapNoBin() {

	string bedLine;
	BED bedEntry;                                                                                                                        
	int lineNum = 0;

	vector<string> bedFields;	// vector of strings for each column in BED file.
	bedFields.reserve(6);		// reserve space for worst case (BED 6)

	// Case 1: Proper BED File.
	if ( (this->bedFile != "") && (this->bedFile != "stdin") ) {

		// open the BED file for reading                                                                                                                                      
		ifstream bed(bedFile.c_str(), ios::in);
		if ( !bed ) {
			cerr << "Error: The requested bed file (" <<bedFile << ") could not be opened. Exiting!" << endl;
			exit (1);
		}

		while (getline(bed, bedLine)) {

			if ((bedLine.find_first_of("track") == 1) || (bedLine.find_first_of("browser") == 1)) {
				continue;
			}
			else {
				Tokenize(bedLine,bedFields);

				lineNum++;

				if (parseBedLine(bedEntry, bedFields, lineNum)) {
					bedEntry.count = 0;
					this->bedMapNoBin[bedEntry.chrom].push_back(bedEntry);	
				}
				bedFields.clear();
			}
		}
	}
	// Case 2: STDIN.
	else {
				
		while (getline(cin, bedLine)) {

			if ((bedLine.find_first_of("track") == 1) || (bedLine.find_first_of("browser") == 1)) {
				continue;
			}
			else {
				vector<string> bedFields;
				Tokenize(bedLine,bedFields);

				lineNum++;

				if (parseBedLine(bedEntry, bedFields, lineNum)) {
					bedEntry.count = 0;
					this->bedMapNoBin[bedEntry.chrom].push_back(bedEntry);	
				}
				bedFields.clear();
			}
		}
	}

	// sort the BED entries for each chromosome
	// in ascending order of start position
	for (masterBedMapNoBin::iterator m = this->bedMapNoBin.begin(); m != this->bedMapNoBin.end(); ++m) {
		sort(m->second.begin(), m->second.end(), sortByStart);		
	}
}


/*
	reportBed
	
	Writes the _original_ BED entry.
	Works for BED3 - BED6.
*/
void BedFile::reportBed(const BED &bed) {
	
	if (this->bedType == 3) {
		cout << bed.chrom << "\t" << bed.start << "\t" << bed.end;
	}
	else if (this->bedType == 4) {
		cout << bed.chrom << "\t" << bed.start << "\t" << bed.end << "\t"
		<< bed.name;
	}
	else if (this->bedType == 5) {
		cout << bed.chrom << "\t" << bed.start << "\t" << bed.end << "\t"
		<< bed.name << "\t" << bed.score;
	}
	else if (this->bedType == 6) {
		cout << bed.chrom << "\t" << bed.start << "\t" << bed.end << "\t" 
		<< bed.name << "\t" << bed.score << "\t" << bed.strand;
	}
}


/*
	reportBedRange
	
	Writes a custom start->end for a BED entry.
	Works for BED3 - BED6.
*/
void BedFile::reportBedRange(const BED &bed, int &start, int &end) {

	if (this->bedType == 3) {
		cout << bed.chrom << "\t" << start << "\t" << end;
	}
	else if (this->bedType == 4) {
		cout << bed.chrom << "\t" << start << "\t" << end << "\t"
		<< bed.name;
	}
	else if (this->bedType == 5) {
		cout << bed.chrom << "\t" << start << "\t" << end << "\t"
		<< bed.name << "\t" << bed.score;
	}
	else if (this->bedType == 6) {
		cout << bed.chrom << "\t" << start << "\t" << end << "\t" 
		<< bed.name << "\t" << bed.score << "\t" << bed.strand;
	}
	
}


