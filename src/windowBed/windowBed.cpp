/*****************************************************************************
  windowBed.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "lineFileUtilities.h"
#include "windowBed.h"


/*
	Constructor
*/
BedWindow::BedWindow(string &bedAFile, string &bedBFile, int &leftSlop, int &rightSlop, bool &anyHit, bool &noHit, 
					bool &writeCount, bool &strandWindows, bool &matchOnStrand) {

	this->bedAFile = bedAFile;
	this->bedBFile = bedBFile;

	this->leftSlop = leftSlop;
	this->rightSlop = rightSlop;

	this->anyHit = anyHit;
	this->noHit = noHit;
	this->writeCount = writeCount;
	this->strandWindows = strandWindows;	
	this->matchOnStrand = matchOnStrand;
		
	this->bedA = new BedFile(bedAFile);
	this->bedB = new BedFile(bedBFile);
}



/*
	Destructor
*/
BedWindow::~BedWindow(void) {
}



void BedWindow::FindWindowOverlaps(BED &a, vector<BED> &hits) {
	
	// update the current feature's start and end
	// according to the slop requested (slop = 0 by default)
	int aFudgeStart = 0;
	int aFudgeEnd;


	// Does the user want to treat the windows based on strand?
	// If so, 
	// if "+", then left is left and right is right
	// if "-", the left is right and right is left.
	if (this->strandWindows) {
		
		if (a.strand == "+") {
			if ((a.start - this->leftSlop) > 0) {
				aFudgeStart = a.start - this->leftSlop;
			}
			else {
				aFudgeStart = 0;
			}
			aFudgeEnd = a.end + this->rightSlop;
		}
		else {
			if ((a.start - this->rightSlop) > 0) {
				aFudgeStart = a.start - this->rightSlop;
			}
			else {
				aFudgeStart = 0;
			}
			aFudgeEnd = a.end + this->leftSlop;
		}
	}
	else {
		if ((a.start - this->leftSlop) > 0) {
			aFudgeStart = a.start - this->leftSlop;
		}
		else {
			aFudgeStart = 0;
		}
		aFudgeEnd = a.end + this->rightSlop;
	}
	
	bedB->binKeeperFind(bedB->bedMap[a.chrom], aFudgeStart, aFudgeEnd, hits);

	int numOverlaps = 0;
	for (vector<BED>::iterator h = hits.begin(); h != hits.end(); ++h) {
	
		// if forcing strandedness, move on if the hit
		// is not on the same strand as A.
		if ((this->matchOnStrand) && (a.strand != h->strand)) {
			continue;		// continue force the next iteration of the for loop.
		}
	
		int s = max(aFudgeStart, h->start);
		int e = min(aFudgeEnd, h->end);
	
		if (s < e) {
			// is there enough overlap (default ~ 1bp)
			if ( ((float)(e-s) / (float)(a.end - a.start)) > 0 ) { 
				numOverlaps++;	
				if (!anyHit && !noHit && !writeCount) {			
					bedA->reportBedTab(a);
					bedB->reportBedNewLine(*h);
				}
			}
		}
	}
	if (anyHit && (numOverlaps >= 1)) {
		bedA->reportBedNewLine(a);	}
	else if (writeCount) {
		bedA->reportBedTab(a); printf("\t%d\n", numOverlaps);
	}
	else if (noHit && (numOverlaps == 0)) {
		bedA->reportBedNewLine(a);
	}
}

 
void BedWindow::WindowIntersectBed(istream &bedInput) {

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
		
	BED a;
	// process each entry in A
	while (getline(bedInput, bedLine)) {

		lineNum++;
		Tokenize(bedLine,bedFields);
	
		// find the overlaps with B if it's a valid BED entry. 
		if (bedA->parseLine(a, bedFields, lineNum)) {
			FindWindowOverlaps(a, hits);
			hits.clear();
		}
		
		// reset for the next input line
		bedFields.clear();
	}
}
// END WindowIntersectBed


void BedWindow::DetermineBedInput() {
	if (bedA->bedFile != "stdin") {   // process a file
		ifstream beds(bedA->bedFile.c_str(), ios::in);
		if ( !beds ) {
			cerr << "Error: The requested bed file (" << bedA->bedFile << ") could not be opened. Exiting!" << endl;
			exit (1);
		}
		WindowIntersectBed(beds);
	}
	else {   						// process stdin
		WindowIntersectBed(cin);		
	}
}

