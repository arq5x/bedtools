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

	_bedAFile      = bedAFile;
	_bedBFile      = bedBFile;

	_leftSlop      = leftSlop;
	_rightSlop     = rightSlop;

	_anyHit        = anyHit;
	_noHit         = noHit;
	_writeCount    = writeCount;
	_strandWindows = strandWindows;	
	_matchOnStrand = matchOnStrand;
		
	_bedA          = new BedFile(bedAFile);
	_bedB          = new BedFile(bedBFile);
	
	// do the work
	WindowIntersectBed();
}



/*
	Destructor
*/
BedWindow::~BedWindow(void) {
}



void BedWindow::FindWindowOverlaps(BED &a, vector<BED> &hits) {
	
	/* 
		Adjust the start and end of a based on the requested window
	*/

	// update the current feature's start and end
	// according to the slop requested (slop = 0 by default)
	int aFudgeStart = 0;
	int aFudgeEnd;

	// Does the user want to treat the windows based on strand?
	// If so, 
	// if "+", then left is left and right is right
	// if "-", the left is right and right is left.
	if (_strandWindows) {
		if (a.strand == "+") {
			if ((a.start - _leftSlop) > 0) aFudgeStart = a.start - _leftSlop;
			else aFudgeStart = 0;
			aFudgeEnd = a.end + _rightSlop;
		}
		else {
			if ((a.start - _rightSlop) > 0) aFudgeStart = a.start - _rightSlop;
			else aFudgeStart = 0;
			aFudgeEnd = a.end + _leftSlop;
		}
	}
	// If not, add the windows irrespective of strand
	else {
		if ((a.start - _leftSlop) > 0) aFudgeStart = a.start - _leftSlop;
		else aFudgeStart = 0;
		aFudgeEnd = a.end + _rightSlop;
	}
	
	
	/* 
		Now report the hits (if any) based on the window around a.
	*/
	// get the hits in B for the A feature
	_bedB->FindOverlapsPerBin(a.chrom, aFudgeStart, aFudgeEnd, a.strand, hits, _matchOnStrand);

	int numOverlaps = 0;
	
	// loop through the hits and report those that meet the user's criteria
	vector<BED>::const_iterator h = hits.begin();
	vector<BED>::const_iterator hitsEnd = hits.end();
	for (; h != hitsEnd; ++h) {
	
		int s = max(aFudgeStart, h->start);
		int e = min(aFudgeEnd, h->end);
		int overlapBases = (e - s);				// the number of overlapping bases b/w a and b
		int aLength = (a.end - a.start);		// the length of a in b.p.
			
		if (s < e) {
			// is there enough overlap (default ~ 1bp)
			if ( ((float) overlapBases / (float) aLength) > 0 ) { 
				numOverlaps++;	
				if (_anyHit == false && _noHit == false && _writeCount == false) {			
					_bedA->reportBedTab(a);
					_bedB->reportBedNewLine(*h);
				}
			}
		}
	}
	if (_anyHit == true && (numOverlaps >= 1)) {
		_bedA->reportBedNewLine(a);	}
	else if (_writeCount == true) {
		_bedA->reportBedTab(a); printf("\t%d\n", numOverlaps);
	}
	else if (_noHit == true && (numOverlaps == 0)) {
		_bedA->reportBedNewLine(a);
	}
}

 
void BedWindow::WindowIntersectBed() {

	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	_bedB->loadBedFileIntoMap();

	BED a;                                                                                                                    
	int lineNum = 0;					// current input line number
	vector<BED> hits;					// vector of potential hits
	hits.reserve(100);

	// process each entry in A
	_bedA->Open();
	while (_bedA->GetNextBed(a, lineNum)) {
		FindWindowOverlaps(a, hits);
		hits.clear();
	}
	_bedA->Close();
}


