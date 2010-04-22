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
	
	unsigned int aLen = a.end - a.start;
	unsigned int bLen = b.end - b.start;
	
	if (aLen < bLen) return true;
	else return false;
};

bool sortBySizeDesc(const BED &a, const BED &b) {
	
	unsigned int aLen = a.end - a.start;
	unsigned int bLen = b.end - b.start;
	
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
	NOTE: Taken ~verbatim from kent source.
	
	Given start,end in chromosome coordinates assign it
	* a bin.   There's a bin for each 128k segment, for each
	* 1M segment, for each 8M segment, for each 64M segment,
	* and for each chromosome (which is assumed to be less than
	* 512M.)  A range goes into the smallest bin it will fit in. 
*/
int getBin(int start, int end) {
	int startBin = start;
	int endBin = end-1;
	startBin >>= _binFirstShift;
	endBin >>= _binFirstShift;
	
	for (int i = 0; i < 6; ++i) {
		if (startBin == endBin) {
			return binOffsetsExtended[i] + startBin;
		}
		startBin >>= _binNextShift;
		endBin >>= _binNextShift;
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
	delete _bedStream;
}


BedLineStatus BedFile::GetNextBed (BED &bed, int &lineNum) {

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


void BedFile::FindOverlapsPerBin(string chrom, int start, int end, string strand, vector<BED> &hits, bool forceStrand) {

	int startBin, endBin;
	startBin = (start >> _binFirstShift);
	endBin = ((end-1) >> _binFirstShift);

	// loop through each bin "level" in the binning hierarchy
	for (int i = 0; i < 6; ++i) {
		
		// loop through each bin at this level of the hierarchy
		int offset = binOffsetsExtended[i];
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


bool BedFile::FindOneOrMoreOverlapsPerBin(string chrom, int start, int end, string strand, 
	bool forceStrand, float overlapFraction) {

	int startBin, endBin;
	startBin = (start >> _binFirstShift);
	endBin = ((end-1) >> _binFirstShift);

	int aLength = (end - start);
	
	// loop through each bin "level" in the binning hierarchy
	for (int i = 0; i < 6; ++i) {
		
		// loop through each bin at this level of the hierarchy
		int offset = binOffsetsExtended[i];
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


bool BedFile::FindOneOrMoreReciprocalOverlapsPerBin(string chrom, int start, int end, string strand, 
	bool forceStrand, float overlapFraction) {

	int startBin, endBin;
	startBin = (start >> _binFirstShift);
	endBin = ((end-1) >> _binFirstShift);

	int aLength = (end - start);
	
	// loop through each bin "level" in the binning hierarchy
	for (int i = 0; i < 6; ++i) {
		
		// loop through each bin at this level of the hierarchy
		int offset = binOffsetsExtended[i];
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
	for (int i = 0; i < 6; ++i) {
	
		// loop through each bin at this level of the hierarchy	
		int offset = binOffsetsExtended[i];
		for (int j = (startBin+offset); j <= (endBin+offset); ++j) {
			
			// loop through each feature in this chrom/bin and see if it overlaps
			// with the feature that was passed in.  if so, add the feature to 
			// the list of hits.
			vector<BED>::iterator bedItr = bedMap[a.chrom][j].begin();
			vector<BED>::iterator bedEnd = bedMap[a.chrom][j].end();		
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


BedLineStatus BedFile::parseLine (BED &bed, const vector<string> &lineVector, int &lineNum) {
		
	char *p2End, *p3End, *p4End, *p5End;
	long l2, l3, l4, l5;
	
	// bail out if we have a blank line
	if (lineVector.size() == 0) 
		return BED_BLANK;
	
	if ((lineVector[0].find("track") == string::npos) && (lineVector[0].find("browser") == string::npos) && (lineVector[0].find("#") == string::npos) ) {

		// we need at least 3 columns
		if (lineVector.size() >= 3) {

			// test if columns	2 and 3 are integers.  If so, assume BED.
			l2 = strtol(lineVector[1].c_str(), &p2End, 10);
			l3 = strtol(lineVector[2].c_str(), &p3End, 10);
			
			// strtol  will set p2End or p3End to the start of the string if non-integral, base 10
			if (p2End != lineVector[1].c_str() && p3End != lineVector[2].c_str()) {
				setGff(false);
				if (parseBedLine(bed, lineVector, lineNum) == true)
					return BED_VALID;
			}
			else if (lineVector.size() == 9) {
				// otherwise test if columns 4 and 5 are integers.  If so, assume GFF.
				l4 = strtol(lineVector[3].c_str(), &p4End, 10);
				l5 = strtol(lineVector[4].c_str(), &p5End, 10);

				// strtol  will set p4End or p5End to the start of the string if non-integral, base 10
				if (p4End != lineVector[3].c_str() && p5End != lineVector[4].c_str()) {		
					setGff(true);
					if (parseGffLine(bed, lineVector, lineNum) == true)
						return BED_VALID;
				}
			}
			else {
				cerr << "Unexpected file format.  Please use tab-delimited BED or GFF" << endl;
				exit(1);
			}
		}
		// gripe if it's not a blank line	
		else if (lineVector.size() != 0) {
			cerr << "It looks as though you have less than 3 columns.  Are you sure your files are tab-delimited?" << endl;
			exit(1);
		}
	}
	else {
		lineNum--;
		return BED_HEADER;	
	}
	
	// default
	return BED_INVALID;
}



bool BedFile::parseBedLine (BED &bed, const vector<string> &lineVector, int lineNum) {

	if ( (lineNum > 1) && (lineVector.size() == this->bedType)) {

		if (this->bedType == 3) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = "";
			bed.score = "";
			bed.strand = "";
		}
		else if (this->bedType == 4) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = "";
			bed.strand = "";
		}
		else if (this->bedType ==5) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = lineVector[4];
			bed.strand = "";		
		}
		else if (this->bedType == 6) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = lineVector[4];
			bed.strand = lineVector[5];
		}
		else if (this->bedType > 6) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = lineVector[4];
			bed.strand = lineVector[5];
			
			for (unsigned int i = 6; i < lineVector.size(); ++i) {
				bed.otherFields.push_back(lineVector[i]); 
			}
		}
		else {
			cerr << "Error: unexpected number of fields at line: " << lineNum << ".  Verify that your files are TAB-delimited and that your BED file has 3,4,5 or 6 fields.  Exiting..." << endl;
			exit(1);
		}
		
		if (bed.start > bed.end) {
			cerr << "Error: malformed BED entry at line " << lineNum << ". Start was greater than end. Exiting." << endl;
			exit(1);
		}
		else if ( (bed.start < 0) || (bed.end < 0) ) {
			cerr << "Error: malformed BED entry at line " << lineNum << ". Coordinate detected that is < 0. Exiting." << endl;
			exit(1);
		}
		else return true;
	}
	else if ((lineNum == 1) && (lineVector.size() >= 3)) {
		this->bedType = lineVector.size();
		if (this->bedType == 3) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = "";
			bed.score = "";
			bed.strand = "";
		}
		else if (this->bedType == 4) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = "";
			bed.strand = "";
		}
		else if (this->bedType ==5) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = lineVector[4];
			bed.strand = "";
		}
		else if (this->bedType == 6) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = lineVector[4];
			bed.strand = lineVector[5];
		}
		else if (this->bedType > 6) {
			bed.chrom = lineVector[0];
			bed.start = atoi(lineVector[1].c_str());
			bed.end = atoi(lineVector[2].c_str());
			bed.name = lineVector[3];
			bed.score = lineVector[4];
			bed.strand = lineVector[5];
			
			for (unsigned int i = 6; i < lineVector.size(); ++i) {
				bed.otherFields.push_back(lineVector[i]); 
			}
		}
		else {
			cerr << "Error: unexpected number of fields at line: " << lineNum << ".  Verify that your files are TAB-delimited and that your BED file has 3,4,5 or 6 fields.  Exiting..." << endl;
			exit(1);
		}
		
		if (bed.start > bed.end) {
			cerr << "Error: malformed BED entry at line " << lineNum << ". Start was greater than end. Exiting." << endl;
			exit(1);
		}
		else if ( (bed.start < 0) || (bed.end < 0) ) {
			cerr << "Error: malformed BED entry at line " << lineNum << ". Coordinate detected that is < 0. Exiting." << endl;
			exit(1);
		}
		else return true;
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



bool BedFile::parseGffLine (BED &bed, const vector<string> &lineVector, int lineNum) {

/*
1.  seqname - The name of the sequence. Must be a chromosome or scaffold.
2. source - The program that generated this feature.
3. feature - The name of this type of feature. Some examples of standard feature types are "CDS", 
			"start_codon", "stop_codon", and "exon".
4. start - The starting position of the feature in the sequence. The first base is numbered 1.
5. end - The ending position of the feature (inclusive).
6. score - A score between 0 and 1000. If the track line useScore attribute is set to 1 
			for this annotation data set, the score value will determine the level of gray 
			in which this feature is displayed (higher numbers = darker gray). 
			If there is no score value, enter ".".
7. strand - Valid entries include '+', '-', or '.' (for don't know/don't care).
8. frame - If the feature is a coding exon, frame should be a number between 0-2 that 
			represents the reading frame of the first base. If the feature is not a coding exon, 
			the value should be '.'.
9. group - All lines with the same group are linked together into a single item.
*/
	if ( (lineNum > 1) && (lineVector.size() == this->bedType)) {
		
		if (this->bedType == 9 && _isGff) {
			bed.chrom = lineVector[0];
			// substract 1 to force the start to be BED-style
			bed.start = atoi(lineVector[3].c_str()) - 1;
			bed.end = atoi(lineVector[4].c_str());
			bed.name = lineVector[2];
			bed.score = lineVector[5];
			bed.strand = lineVector[6];
			bed.otherFields.push_back(lineVector[1]);  // add GFF "source". unused in BED
			bed.otherFields.push_back(lineVector[7]);  // add GFF "fname". unused in BED
			bed.otherFields.push_back(lineVector[8]);  // add GFF "group". unused in BED
		}
		else {
			cerr << "Error: unexpected number of fields at line: " << lineNum << 
					".  Verify that your files are TAB-delimited and that your GFF file has 9 fields.  Exiting..." << endl;
			exit(1);
		}
		
		if (bed.start > bed.end) {
			cerr << "Error: malformed GFF entry at line " << lineNum << ". Start was greater than end. Exiting." << endl;
			exit(1);
		}
		else if ( (bed.start < 0) || (bed.end < 0) ) {
			cerr << "Error: malformed GFF entry at line " << lineNum << ". Coordinate detected that is < 1. Exiting." << endl;
			exit(1);
		}
		else return true;
	}
	else if ((lineNum == 1) && (lineVector.size() == 9)) {
		this->bedType = lineVector.size();
		
		if (this->bedType == 9 && _isGff) {
			bed.chrom = lineVector[0];
			// substract 1 to force the start to be BED-style
			bed.start = atoi(lineVector[3].c_str()) - 1;
			bed.end = atoi(lineVector[4].c_str());
			bed.name = lineVector[2];
			bed.score = lineVector[5];
			bed.strand = lineVector[6];
		
			bed.otherFields.push_back(lineVector[1]);  // add GFF "source". unused in BED
			bed.otherFields.push_back(lineVector[7]);  // add GFF "fname". unused in BED
			bed.otherFields.push_back(lineVector[8]);  // add GFF "group". unused in BED
		
			return true;
		}
		else {
			cout << this->bedType << " " << _isGff << endl;
			cerr << "Error: unexpected number of fields at line: " << lineNum << 
					".  Verify that your files are TAB-delimited and that your GFF file has 9 fields.  Exiting..." << endl;
			exit(1);
		}
		
		if (bed.start > bed.end) {
			cerr << "Error: malformed GFF entry at line " << lineNum << ". Start was greater than end. Exiting." << endl;
			exit(1);
		}
		else if ( (bed.start < 0) || (bed.end < 0) ) {
			cerr << "Error: malformed GFF entry at line " << lineNum << ". Coordinate detected that is < 1. Exiting." << endl;
			exit(1);
		}
		else return true;
	}
	else if (lineVector.size() == 1) {
		cerr << "Only one GFF field detected: " << lineNum << ".  Verify that your files are TAB-delimited.  Exiting..." << endl;
		exit(1);		
	}
	else if ((lineVector.size() != this->bedType) && (lineVector.size() != 0)) {
		cerr << "Differing number of GFF fields encountered at line: " << lineNum << ".  Exiting..." << endl;
		exit(1);
	}
	else if ((lineVector.size() < 9) && (lineVector.size() != 0)) {
		cerr << "TAB delimited GFF file with 9 fields is required at line: "<< lineNum << ".  Exiting..." << endl;
		exit(1);
	}
	return false;
}


void BedFile::loadBedFileIntoMap() {

	BED bedEntry, nullBed;
	int lineNum = 0;
	BedLineStatus bedStatus;
		
	Open();
	bedStatus = this->GetNextBed(bedEntry, lineNum);	
	while (bedStatus != BED_INVALID) {
		// ignore headers and blank lines
		if (bedStatus == BED_VALID) {		
			int bin = getBin(bedEntry.start, bedEntry.end);
			bedEntry.count = 0;
			bedEntry.minOverlapStart = INT_MAX;
			bedMap[bedEntry.chrom][bin].push_back(bedEntry);

			bedEntry = nullBed;
		}
		bedStatus = this->GetNextBed(bedEntry, lineNum);	
	}
	Close();
}


void BedFile::loadBedFileIntoMapNoBin() {
	
	BED bedEntry, nullBed;
	int lineNum = 0;
	BedLineStatus bedStatus;
	
	Open();
	bedStatus = this->GetNextBed(bedEntry, lineNum);
	while (bedStatus != BED_INVALID) {
		// ignore headers and blank lines
		if (bedStatus == BED_VALID) {
			bedEntry.count = 0;
			bedEntry.minOverlapStart = INT_MAX;
			bedMapNoBin[bedEntry.chrom].push_back(bedEntry);
		
			bedEntry = nullBed;	
		}
		bedStatus = this->GetNextBed(bedEntry, lineNum);	
	}
	Close();
}


/*
	reportBedTab
	
	Writes the _original_ BED entry with a TAB
	at the end of the line.
	Works for BED3 - BED6.
*/
void BedFile::reportBedTab(const BED &bed) {
	
	if (_isGff == false) {
		if (this->bedType == 3) {
			printf ("%s\t%d\t%d\t", bed.chrom.c_str(), bed.start, bed.end);
		}
		else if (this->bedType == 4) {
			printf ("%s\t%d\t%d\t%s\t", bed.chrom.c_str(), bed.start, bed.end, bed.name.c_str());
		}
		else if (this->bedType == 5) {
			printf ("%s\t%d\t%d\t%s\t%s\t", bed.chrom.c_str(), bed.start, bed.end, bed.name.c_str(), 
											bed.score.c_str());
		}
		else if (this->bedType == 6) {
			printf ("%s\t%d\t%d\t%s\t%s\t%s\t", bed.chrom.c_str(), bed.start, bed.end, bed.name.c_str(), 
												bed.score.c_str(), bed.strand.c_str());
		}
		else if (this->bedType > 6) {
			printf ("%s\t%d\t%d\t%s\t%s\t%s\t", bed.chrom.c_str(), bed.start, bed.end, bed.name.c_str(), 
												bed.score.c_str(), bed.strand.c_str());
		
			vector<string>::const_iterator othIt = bed.otherFields.begin(); 
			vector<string>::const_iterator othEnd = bed.otherFields.end(); 
			for ( ; othIt != othEnd; ++othIt) {
				printf("%s\t", othIt->c_str());
			}
		}
	}	
	else if (this->bedType == 9) {
		printf ("%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t", bed.chrom.c_str(), bed.otherFields[0].c_str(),
														 bed.name.c_str(), bed.start+1, bed.end, 
														 bed.score.c_str(), bed.strand.c_str(),
														 bed.otherFields[1].c_str(), bed.otherFields[2].c_str());

	}
}



/*
	reportBedNewLine
	
	Writes the _original_ BED entry with a NEWLINE
	at the end of the line.
	Works for BED3 - BED6.
*/
void BedFile::reportBedNewLine(const BED &bed) {
	
	if (_isGff == false) {
		if (this->bedType == 3) {
			printf ("%s\t%d\t%d\n", bed.chrom.c_str(), bed.start, bed.end);
		}
		else if (this->bedType == 4) {
			printf ("%s\t%d\t%d\t%s\n", bed.chrom.c_str(), bed.start, bed.end, bed.name.c_str());
		}
		else if (this->bedType == 5) {
			printf ("%s\t%d\t%d\t%s\t%s\n", bed.chrom.c_str(), bed.start, bed.end, bed.name.c_str(), 
											bed.score.c_str());
		}
		else if (this->bedType == 6) {
			printf ("%s\t%d\t%d\t%s\t%s\t%s\n", bed.chrom.c_str(), bed.start, bed.end, bed.name.c_str(), 
												bed.score.c_str(), bed.strand.c_str());
		}
		else if (this->bedType > 6) {
			printf ("%s\t%d\t%d\t%s\t%s\t%s\t", bed.chrom.c_str(), bed.start, bed.end, bed.name.c_str(), 
												bed.score.c_str(), bed.strand.c_str());
		
			vector<string>::const_iterator othIt = bed.otherFields.begin(); 
			vector<string>::const_iterator othEnd = bed.otherFields.end(); 
			for ( ; othIt != othEnd; ++othIt) {
				printf("%s\t", othIt->c_str());
			}
			printf("\n");
		}
	}
	else if (this->bedType == 9) {
		printf ("%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\n", bed.chrom.c_str(), bed.otherFields[0].c_str(),
														 bed.name.c_str(), bed.start+1, bed.end, 
														 bed.score.c_str(), bed.strand.c_str(),
														 bed.otherFields[1].c_str(), bed.otherFields[2].c_str());

	}
}



/*
	reportBedRangeNewLine
	
	Writes a custom start->end for a BED entry
	with a NEWLINE at the end of the line.

	Works for BED3 - BED6.
*/
void BedFile::reportBedRangeTab(const BED &bed, int start, int end) {

	if (_isGff == false) {
		if (this->bedType == 3) {
			printf ("%s\t%d\t%d\t", bed.chrom.c_str(), start, end);
		}
		else if (this->bedType == 4) {
			printf ("%s\t%d\t%d\t%s\t", bed.chrom.c_str(), start, end, bed.name.c_str());
		}
		else if (this->bedType == 5) {
			printf ("%s\t%d\t%d\t%s\t%s\t", bed.chrom.c_str(), start, end, bed.name.c_str(), 
											bed.score.c_str());
		}
		else if (this->bedType == 6) {
			printf ("%s\t%d\t%d\t%s\t%s\t%s\t", bed.chrom.c_str(), start, end, bed.name.c_str(), 
												bed.score.c_str(), bed.strand.c_str());
		}
		else if (this->bedType > 6) {
			printf ("%s\t%d\t%d\t%s\t%s\t%s\t", bed.chrom.c_str(), start, end, bed.name.c_str(), 
												bed.score.c_str(), bed.strand.c_str());
		
			vector<string>::const_iterator othIt = bed.otherFields.begin(); 
			vector<string>::const_iterator othEnd = bed.otherFields.end(); 
			for ( ; othIt != othEnd; ++othIt) {
				printf("%s\t", othIt->c_str());
			}
		}
	}
	else if (this->bedType == 9) {
		printf ("%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t", bed.chrom.c_str(), bed.otherFields[0].c_str(),
														 bed.name.c_str(), start+1, end, 
														 bed.score.c_str(), bed.strand.c_str(),
														 bed.otherFields[1].c_str(), bed.otherFields[2].c_str());

	}
}



/*
	reportBedRangeTab
	
	Writes a custom start->end for a BED entry 
	with a TAB at the end of the line.
	
	Works for BED3 - BED6.
*/
void BedFile::reportBedRangeNewLine(const BED &bed, int start, int end) {

	if (_isGff == false) {
		if (this->bedType == 3) {
			printf ("%s\t%d\t%d\n", bed.chrom.c_str(), start, end);
		}
		else if (this->bedType == 4) {
			printf ("%s\t%d\t%d\t%s\n", bed.chrom.c_str(), start, end, bed.name.c_str());
		}
		else if (this->bedType == 5) {
			printf ("%s\t%d\t%d\t%s\t%s\n", bed.chrom.c_str(), start, end, bed.name.c_str(), 
											bed.score.c_str());
		}
		else if (this->bedType == 6) {
			printf ("%s\t%d\t%d\t%s\t%s\t%s\n", bed.chrom.c_str(), start, end, bed.name.c_str(), 
												bed.score.c_str(), bed.strand.c_str());
		}
		else if (this->bedType > 6) {
			printf ("%s\t%d\t%d\t%s\t%s\t%s\t", bed.chrom.c_str(), start, end, bed.name.c_str(), 
												bed.score.c_str(), bed.strand.c_str());
		
			vector<string>::const_iterator othIt = bed.otherFields.begin(); 
			vector<string>::const_iterator othEnd = bed.otherFields.end(); 
			for ( ; othIt != othEnd; ++othIt) {
				printf("%s\t", othIt->c_str());
			}
			printf("\n");
		}
	}
	else if (this->bedType == 9) {	// add 1 to the start for GFF
		printf ("%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\n", bed.chrom.c_str(), bed.otherFields[0].c_str(),
														 bed.name.c_str(), start+1, end, 
														 bed.score.c_str(), bed.strand.c_str(),
														 bed.otherFields[1].c_str(), bed.otherFields[2].c_str());

	}
}


/*
	reportNullBedTab
*/
void BedFile::reportNullBedTab() {
	
	if (_isGff == false) {
		if (this->bedType == 3) {
			printf (".\t-1\t-1\t");
		}
		else if (this->bedType == 4) {
			printf (".\t-1\t-1\t.\t");
		}
		else if (this->bedType == 5) {
			printf (".\t-1\t-1\t.\t-1\t");
		}
		else if (this->bedType == 6) {
			printf (".\t-1\t-1\t.\t-1\t.\t");
		}
		else if (this->bedType > 6) {
			printf (".\t-1\t-1\t.\t-1\t.\t");
			for (unsigned int i = 6; i < this->bedType; ++i) {
				printf(".\t");
			}
		}
	}	
	else if (this->bedType == 9) {
		printf (".\t.\t.\t-1\t-1\t-1\t.\t.\t.\t");
	}
}


/*
	reportNullBedTab
*/
void BedFile::reportNullBedNewLine() {
	
	if (_isGff == false) {
		if (this->bedType == 3) {
			printf (".\t-1\t-1\n");
		}
		else if (this->bedType == 4) {
			printf (".\t-1\t-1\t.\n");
		}
		else if (this->bedType == 5) {
			printf (".\t-1\t-1\t.\t-1\n");
		}
		else if (this->bedType == 6) {
			printf (".\t-1\t-1\t.\t-1\t.\n");
		}
		else if (this->bedType > 6) {
			printf (".\t-1\t-1\t.\t-1\t.\n");
			for (unsigned int i = 6; i < this->bedType; ++i) {
				printf(".\t");
			}
			printf("\n");
		}
	}	
	else if (this->bedType == 9) {
		printf (".\t.\t.\t-1\t-1\t-1\t.\t.\t.\n");
	}
}

