/*****************************************************************************
  bedFile.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licensed under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "lineFileUtilities.h"
#include "bedFile.h"


/***********************************************
Sorting comparison functions
************************************************/
bool sortByChrom(BED const & a, BED const & b) {
	if (a.chrom < b.chrom) return true;
	else return false;
};

bool sortByStart(const BED &a, const BED &b) {
	if (a.start < b.start) return true;
	else return false;
};

bool sortBySizeAsc(const BED &a, const BED &b) {
	
	CHRPOS aLen = a.end - a.start;
	CHRPOS bLen = b.end - b.start;
	
	if (aLen < bLen) return true;
	else return false;
};

bool sortBySizeDesc(const BED &a, const BED &b) {
	
	CHRPOS aLen = a.end - a.start;
	CHRPOS bLen = b.end - b.start;
	
	if (aLen > bLen) return true;
	else return false;
};

bool sortByScoreAsc(const BED &a, const BED &b) {
	if (a.score < b.score) return true;
	else return false;
};

bool sortByScoreDesc(const BED &a, const BED &b) {
	if (a.score > b.score) return true;
	else return false;
};


bool byChromThenStart(BED const & a, BED const & b) {

	if (a.chrom < b.chrom) return true;
	else if (a.chrom > b.chrom) return false;

	if (a.start < b.start) return true;
	else if (a.start >= b.start) return false;

	return false;
};



/* 
	NOTE: Tweaked from kent source.
*/
int getBin(CHRPOS start, CHRPOS end) {
    --end;
    start >>= _binFirstShift;
    end   >>= _binFirstShift;
    
    for (int i = 0; i < _binLevels; ++i) {
        if (start == end) return _binOffsetsExtended[i] + start;
        start >>= _binNextShift;
        end   >>= _binNextShift;
    }
    cerr << "start " << start << ", end " << end << " out of range in findBin (max is 512M)" << endl;
    return 0;
}




/*******************************************
Class methods
*******************************************/

// Constructor
BedFile::BedFile(string &bedFile)
: bedFile(bedFile)
{}

// Destructor
BedFile::~BedFile(void) {
}


void BedFile::Open(void) {
	if (bedFile == "stdin") {
		_bedStream = &cin;
	}
	else {
		size_t foundPos;
	  	foundPos = bedFile.find_last_of(".gz");
		// is this a GZIPPED BED file?
		if (foundPos == bedFile.size() - 1) {
			igzstream beds(bedFile.c_str(), ios::in);
			if ( !beds ) {
				cerr << "Error: The requested bed file (" << bedFile << ") could not be opened. Exiting!" << endl;
				exit (1);
			}
			else {
				// if so, close it (this was just a test)
				beds.close();		
				// now set a pointer to the stream so that we
				// can read the file later on.
				// Thank God for Josuttis, p. 631!
				_bedStream = new igzstream(bedFile.c_str(), ios::in);
			}
		}  
		// not GZIPPED.
		else {
		
			ifstream beds(bedFile.c_str(), ios::in);
			// can we open the file?
			if ( !beds ) {
				cerr << "Error: The requested bed file (" << bedFile << ") could not be opened. Exiting!" << endl;
				exit (1);
			}
			else {
				// if so, close it (this was just a test)
				beds.close();		
				// now set a pointer to the stream so that we
				// can read the file later on.
				// Thank God for Josuttis, p. 631!
				_bedStream = new ifstream(bedFile.c_str(), ios::in);
			}
		}
	}
}


// Close the BED file
void BedFile::Close(void) {
	if (bedFile != "stdin") delete _bedStream;
}


BedLineStatus BedFile::GetNextBed(BED &bed, int &lineNum) {

	// make sure there are still lines to process.
	// if so, tokenize, validate and return the BED entry.
	if (_bedStream->good()) {
		string bedLine;
		vector<string> bedFields;
		bedFields.reserve(12);
		
		// parse the bedStream pointer
		getline(*_bedStream, bedLine);
		lineNum++;

		// split into a string vector.
		Tokenize(bedLine,bedFields);
		
		// load the BED struct as long as it's a valid BED entry.
		return parseLine(bed, bedFields, lineNum);
	}
	
	// default if file is closed or EOF
	return BED_INVALID;
}


void BedFile::FindOverlapsPerBin(string chrom, CHRPOS start, CHRPOS end, char strand, vector<BED> &hits, bool forceStrand) {

	int startBin, endBin;
	startBin = (start >> _binFirstShift);
	endBin = ((end-1) >> _binFirstShift);

	// loop through each bin "level" in the binning hierarchy
	for (int i = 0; i < _binLevels; ++i) {
		
		// loop through each bin at this level of the hierarchy
		int offset = _binOffsetsExtended[i];
		for (int j = (startBin+offset); j <= (endBin+offset); ++j)  {
			
            // loop through each feature in this chrom/bin and see if it overlaps
            // with the feature that was passed in.  if so, add the feature to 
            // the list of hits.
            vector<BED>::const_iterator bedItr = bedMap[chrom][j].begin();
            vector<BED>::const_iterator bedEnd = bedMap[chrom][j].end();
            
            for (; bedItr != bedEnd; ++bedItr) {
                // do we have sufficient overlap?
                if (overlaps(bedItr->start, bedItr->end, start, end) > 0) {
                    // skip the hit if not on the same strand (and we care)
                    if (forceStrand == false) hits.push_back(*bedItr);
                    else if ( (forceStrand == true) && (strand == bedItr->strand)) {
                         hits.push_back(*bedItr);
                    }
                }
            }
		}
		startBin >>= _binNextShift;
		endBin >>= _binNextShift;
	}
}


bool BedFile::FindOneOrMoreOverlapsPerBin(string chrom, CHRPOS start, CHRPOS end, char strand, 
	bool forceStrand, float overlapFraction) {

	int startBin, endBin;
	startBin = (start >> _binFirstShift);
	endBin = ((end-1) >> _binFirstShift);

	int aLength = (end - start);
	
	// loop through each bin "level" in the binning hierarchy
	for (int i = 0; i < _binLevels; ++i) {
		
		// loop through each bin at this level of the hierarchy
		int offset = _binOffsetsExtended[i];
		for (int j = (startBin+offset); j <= (endBin+offset); ++j)  {
			
			// loop through each feature in this chrom/bin and see if it overlaps
			// with the feature that was passed in.  if so, add the feature to 
			// the list of hits.
			vector<BED>::const_iterator bedItr = bedMap[chrom][j].begin();
			vector<BED>::const_iterator bedEnd = bedMap[chrom][j].end();
			for (; bedItr != bedEnd; ++bedItr) {
				int s = max(start, bedItr->start);
				int e = min(end, bedItr->end);
				// the number of overlapping bases b/w a and b
				int overlapBases = (e - s);

				// do we have sufficient overlap?
				if ( (float) overlapBases / (float) aLength  >= overlapFraction) {					
					// skip the hit if not on the same strand (and we care)
					if (forceStrand == false) return true;
					else if ( (forceStrand == true) && (strand == bedItr->strand)) {
						return true;
					}
				}			
			}
		}
		startBin >>= _binNextShift;
		endBin >>= _binNextShift;
	}
	return false;
}


bool BedFile::FindOneOrMoreReciprocalOverlapsPerBin(string chrom, CHRPOS start, CHRPOS end, char strand, 
	bool forceStrand, float overlapFraction) {

	int startBin, endBin;
	startBin = (start >> _binFirstShift);
	endBin = ((end-1) >> _binFirstShift);

	int aLength = (end - start);
	
	// loop through each bin "level" in the binning hierarchy
	for (int i = 0; i < _binLevels; ++i) {
		
		// loop through each bin at this level of the hierarchy
		int offset = _binOffsetsExtended[i];
		for (int j = (startBin+offset); j <= (endBin+offset); ++j)  {
			
			// loop through each feature in this chrom/bin and see if it overlaps
			// with the feature that was passed in.  if so, add the feature to 
			// the list of hits.
			vector<BED>::const_iterator bedItr = bedMap[chrom][j].begin();
			vector<BED>::const_iterator bedEnd = bedMap[chrom][j].end();
			for (; bedItr != bedEnd; ++bedItr) {
				int s = max(start, bedItr->start);
				int e = min(end, bedItr->end);
				// the number of overlapping bases b/w a and b
				int overlapBases = (e - s);
				
				// do we have sufficient overlap?
				if ( (float) overlapBases / (float) aLength  >= overlapFraction) {					
					int bLength = (bedItr->end - bedItr->start);
					float bOverlap = ( (float) overlapBases / (float) bLength );
					if ((forceStrand == false) && (bOverlap >= overlapFraction)) {
						return true;
					}
					else if ( (forceStrand == true) && (strand == bedItr->strand) && (bOverlap >= overlapFraction)) {
						return true;
					}
				}			
			}
		}
		startBin >>= _binNextShift;
		endBin >>= _binNextShift;
	}
	return false;
}


void BedFile::countHits(const BED &a, bool forceStrand) {

    int startBin, endBin;
    startBin = (a.start >> _binFirstShift);
    endBin = ((a.end-1) >> _binFirstShift);

    // loop through each bin "level" in the binning hierarchy	
    for (int i = 0; i < _binLevels; ++i) {

        // loop through each bin at this level of the hierarchy	
        int offset = _binOffsetsExtended[i];
        for (int j = (startBin+offset); j <= (endBin+offset); ++j) {

            // loop through each feature in this chrom/bin and see if it overlaps
            // with the feature that was passed in.  if so, add the feature to 
            // the list of hits.
            vector<BEDCOV>::iterator bedItr = bedCovMap[a.chrom][j].begin();
            vector<BEDCOV>::iterator bedEnd = bedCovMap[a.chrom][j].end();		
            for (; bedItr != bedEnd; ++bedItr) {

                // skip the hit if not on the same strand (and we care)
                if (forceStrand && (a.strand != bedItr->strand)) {
                    continue;
                }
                else if (overlaps(bedItr->start, bedItr->end, a.start, a.end) > 0) {

                    bedItr->count++;
                    bedItr->depthMap[a.start+1].starts++;
                    bedItr->depthMap[a.end].ends++;

                    if (a.start < bedItr->minOverlapStart) {
                        bedItr->minOverlapStart = a.start;
                    }                   
                }
            }
        }
        startBin >>= _binNextShift;
        endBin >>= _binNextShift;
    }
}


void BedFile::setGff (bool gff) {
	if (gff == true) this->_isGff = true;
	else this->_isGff = false;
}


void BedFile::loadBedFileIntoMap() {

	BED bedEntry, nullBed;
	int lineNum = 0;
	BedLineStatus bedStatus;

	Open();
	while ((bedStatus = GetNextBed(bedEntry, lineNum)) != BED_INVALID) {
		if (bedStatus == BED_VALID) {
			int bin = getBin(bedEntry.start, bedEntry.end);
			bedMap[bedEntry.chrom][bin].push_back(bedEntry);
			bedEntry = nullBed;
		}
	}
	Close();
}


void BedFile::loadBedCovFileIntoMap() {

	BED bedEntry, nullBed;
	int lineNum = 0;
	BedLineStatus bedStatus;
		
	Open();
	while ((bedStatus = GetNextBed(bedEntry, lineNum)) != BED_INVALID) {
		if (bedStatus == BED_VALID) {
			int bin = getBin(bedEntry.start, bedEntry.end);
            
            BEDCOV bedCov;
            bedCov.chrom        = bedEntry.chrom;
            bedCov.start        = bedEntry.start;
            bedCov.end          = bedEntry.end;
            bedCov.name         = bedEntry.name;
            bedCov.score        = bedEntry.score;
            bedCov.strand       = bedEntry.strand;
            bedCov.otherFields  = bedEntry.otherFields;
			bedCov.count = 0;
			bedCov.minOverlapStart = INT_MAX;
			
			bedCovMap[bedEntry.chrom][bin].push_back(bedCov);
			bedEntry = nullBed;
		}
	}
	Close();
}


void BedFile::loadBedFileIntoMapNoBin() {
	
	BED bedEntry, nullBed;
	int lineNum = 0;
	BedLineStatus bedStatus;
	
	Open();
	while ((bedStatus = this->GetNextBed(bedEntry, lineNum)) != BED_INVALID) {
		if (bedStatus == BED_VALID) {
			bedMapNoBin[bedEntry.chrom].push_back(bedEntry);
			bedEntry = nullBed;	
		}
	}
	Close();
	
	// sort the BED entries for each chromosome
	// in ascending order of start position
	for (masterBedMapNoBin::iterator m = this->bedMapNoBin.begin(); m != this->bedMapNoBin.end(); ++m) {
		sort(m->second.begin(), m->second.end(), sortByStart);		
	}
}
