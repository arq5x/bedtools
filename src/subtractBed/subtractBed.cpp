/*****************************************************************************
  subtractBed.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "lineFileUtilities.h"
#include "subtractBed.h"


/*
	Constructor
*/
BedSubtract::BedSubtract(string &bedAFile, string &bedBFile, float &overlapFraction, bool &forceStrand) {

	this->bedAFile = bedAFile;
	this->bedBFile = bedBFile;
	this->overlapFraction = overlapFraction;
	this->forceStrand = forceStrand;

	this->bedA = new BedFile(bedAFile);
	this->bedB = new BedFile(bedBFile);
}


/*
	Destructor
*/
BedSubtract::~BedSubtract(void) {
}



void BedSubtract::FindOverlaps(BED &a, vector<BED> &hits) {
	
	// find all of the overlaps between a and B.
	bedB->FindOverlapsPerBin(a.chrom, a.start, a.end, a.strand, hits, this->forceStrand);
	
	//  is A completely spanned by an entry in B?
	//  if so, A should not be reported.
	int numConsumedByB = 0; 
	int numOverlaps = 0;
	vector<BED> bOverlaps;	// list of hits in B.  Special processing if there are multiple.
	
	vector<BED>::const_iterator h = hits.begin();
	vector<BED>::const_iterator hitsEnd = hits.end();
	for (; h != hitsEnd; ++h) {
		
		int s = max(a.start, h->start);
		int e = min(a.end, h->end);		
		int overlapBases = (e - s);				// the number of overlapping bases b/w a and b
		int aLength = (a.end - a.start);		// the length of a in b.p.
			
		if (s < e) {
			
			// is there enough overlap (default ~ 1bp)
			float overlap = ((float) overlapBases / (float) aLength);

			if (overlap >= 1.0) {
				numOverlaps++;
				numConsumedByB++;
			}
			else if ( overlap >= this->overlapFraction ) {
				numOverlaps++;
				bOverlaps.push_back(*h);
			}
		}
	}
	
	if (numOverlaps == 0) {
		// no overlap found, so just report A as-is.
		bedA->reportBedNewLine(a);
	}
	else if (numOverlaps == 1) {
		// one overlap found.  only need to look at the single
		// entry in bOverlaps.
		
		// if A was not "consumed" by any entry in B
		if (numConsumedByB == 0) {
			
			BED theHit = bOverlaps[0];

			// A	++++++++++++
			// B        ----
			// Res. ====    ====					
			if ( (theHit.start > a.start) && (theHit.end < a.end) ) {
				bedA->reportBedRangeNewLine(a,a.start,theHit.start);
				bedA->reportBedRangeNewLine(a,theHit.end,a.end);
			}
			// A	++++++++++++
			// B    ----------
			// Res.           ==        			
			else if (theHit.start == a.start) {
				bedA->reportBedRangeNewLine(a,theHit.end,a.end);
			}	
			// A	      ++++++++++++
			// B    ----------
			// Res.       ====
			else if (theHit.start < a.start) {
				bedA->reportBedRangeNewLine(a,theHit.end,a.end);
			}
			// A	++++++++++++
			// B           ----------
			// Res. =======
			else if (theHit.start > a.start) {
				bedA->reportBedRangeNewLine(a,a.start,theHit.start);	
			}
		}
	}
	else if (numOverlaps > 1) {
		// multiple overlapz found.  look at all the hits
		// and figure out which bases in A survived.  then 
		// report the contigous intervals that survived.
		
		vector<bool> aKeep(a.end - a.start, true);
		
		if (numConsumedByB == 0) {
			// track the number of hit starts and ends at each position in A
			for (vector<BED>::iterator h = bOverlaps.begin(); h != bOverlaps.end(); ++h) {
				int s = max(a.start, h->start);
				int e = min(a.end, h->end);
				
				for (int i = s+1; i <= e; ++i) {
					aKeep[i-a.start-1] = false;
				}
			}
			// report the remaining blocks.
			for (unsigned int i = 0; i < aKeep.size(); ++i) {
				if (aKeep[i] == true) {
					int blockStart = i + a.start;
					while ((aKeep[i] == true) && (i < aKeep.size())) {
						i++;
					}
					int blockEnd = i + a.start;
					blockEnd = min(a.end, blockEnd);
					bedA->reportBedRangeNewLine(a,blockStart,blockEnd);
				}
			}
		}
	}
}

 

void BedSubtract::SubtractBed(istream &bedInput) {

	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	bedB->loadBedFileIntoMap();

	string bedLine;                                                                                                                    
	int lineNum = 0;					// current input line number
	vector<BED> hits;					// vector of potential hits
	vector<string> bedFields;			// vector for a BED entry
	
	// reserve some space
	hits.reserve(100);
	bedFields.reserve(12);	
		
	// process each entry in A
	while (getline(bedInput, bedLine)) {

		lineNum++;
		Tokenize(bedLine,bedFields);
		BED a;
			
		// find the overlaps with B if it's a valid BED entry. 
		if (bedA->parseLine(a, bedFields, lineNum)) {
			FindOverlaps(a, hits);
			hits.clear();
		}
		
		// reset for the next input line
		bedFields.clear();
	}
}
// END Intersect


void BedSubtract::DetermineBedInput() {
	if (bedA->bedFile != "stdin") {   // process a file
		ifstream beds(bedA->bedFile.c_str(), ios::in);
		if ( !beds ) {
			cerr << "Error: The requested bed file (" << bedA->bedFile << ") could not be opened. Exiting!" << endl;
			exit (1);
		}
		SubtractBed(beds);
	}
	else {   						// process stdin
		SubtractBed(cin);		
	}
}

