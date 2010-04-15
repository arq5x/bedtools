/*****************************************************************************
  pairToBed.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "lineFileUtilities.h"
#include "pairToBed.h"


bool IsCorrectMappingForBEDPE (const BamAlignment &bam) {

	if ( (bam.RefID == bam.MateRefID) && (bam.InsertSize > 0) ) {
		return true;
	}
	else if ( (bam.RefID == bam.MateRefID) && (bam.InsertSize == 0) && bam.IsFirstMate() ) {
		return true;
	}
	else if ( (bam.RefID != bam.MateRefID) && bam.IsFirstMate() ) {
		return true;
	}
	else return false;
}


/*
	Constructor
*/

BedIntersectPE::BedIntersectPE(string bedAFilePE, string bedBFile, float overlapFraction, 
						       string searchType, bool forceStrand, bool bamInput, bool bamOutput) {

	_bedAFilePE = bedAFilePE;
	_bedBFile = bedBFile;
	_overlapFraction = overlapFraction;
	_forceStrand = forceStrand;
	_searchType = searchType;
	_bamInput =  bamInput;
	_bamOutput = bamOutput;
	
	_bedA = new BedFilePE(bedAFilePE);
	_bedB = new BedFile(bedBFile);
}


/*
	Destructor
*/

BedIntersectPE::~BedIntersectPE(void) {
}



void BedIntersectPE::FindOverlaps(const BEDPE &a, vector<BED> &hits1, vector<BED> &hits2, const string &type) {

	// list of hits on each end of BEDPE
	// that exceed the requested overlap fraction
	vector<BED> qualityHits1;
	vector<BED> qualityHits2;

	// count of hits on each end of BEDPE
	// that exceed the requested overlap fraction
	int numOverlapsEnd1 = 0;
	int numOverlapsEnd2 = 0;

	// make sure we have a valid chromosome before we search
	if (a.chrom1 != ".") {
		// Find the quality hits between ***end1*** of the BEDPE and the B BED file
		_bedB->FindOverlapsPerBin(a.chrom1, a.start1, a.end1, a.strand1, hits1, _forceStrand);
	
		vector<BED>::const_iterator h = hits1.begin();
		vector<BED>::const_iterator hitsEnd = hits1.end();
		for (; h != hitsEnd; ++h) {
	
			int s = max(a.start1, h->start);
			int e = min(a.end1, h->end);
			int overlapBases = (e - s);				// the number of overlapping bases b/w a and b
			int aLength = (a.end1 - a.start1);		// the length of a in b.p.
		
			// is there enough overlap relative to the user's request? (default ~ 1bp)
			if ( ( (float) overlapBases / (float) aLength ) >= _overlapFraction ) {
				numOverlapsEnd1++;
				
				if (type == "either") {
					_bedA->reportBedPETab(a);
					_bedB->reportBedNewLine(*h);
				}
				else {
					qualityHits1.push_back(*h);
				}	
			}
		}
	}
	
	
	// make sure we have a valid chromosome before we search
	if (a.chrom2 != ".") {	
		// Now find the quality hits between ***end2*** of the BEDPE and the B BED file
		_bedB->FindOverlapsPerBin(a.chrom2, a.start2, a.end2, a.strand2, hits2, _forceStrand);
	
		vector<BED>::const_iterator h = hits2.begin();
		vector<BED>::const_iterator hitsEnd = hits2.end();
		for (; h != hitsEnd; ++h) {
	
			int s = max(a.start2, h->start);
			int e = min(a.end2, h->end);
			int overlapBases = (e - s);				// the number of overlapping bases b/w a and b
			int aLength = (a.end2 - a.start2);		// the length of a in b.p.

			// is there enough overlap relative to the user's request? (default ~ 1bp)
			if ( ( (float) overlapBases / (float) aLength ) >= _overlapFraction ) {
				numOverlapsEnd2++;
				
				if (type == "either") {
					_bedA->reportBedPETab(a);
					_bedB->reportBedNewLine(*h);
				}
				else {
					qualityHits2.push_back(*h);
				}	
			}
		}
	}
	
	// Now report the hits depending on what the user has requested.
	if (type == "neither") {
		if ( (numOverlapsEnd1 == 0) && (numOverlapsEnd2 == 0) ) {
			_bedA->reportBedPENewLine(a); 
		}
	}
	else if (type == "notboth") {
		if ( (numOverlapsEnd1 == 0) && (numOverlapsEnd2 == 0) ) {
			_bedA->reportBedPENewLine(a); 
		}
		else if ( (numOverlapsEnd1 > 0) && (numOverlapsEnd2 == 0) ) {
			for (vector<BED>::iterator q = qualityHits1.begin(); q != qualityHits1.end(); ++q) {
				_bedA->reportBedPETab(a);
				_bedB->reportBedNewLine(*q);
			}
		}
		else if ( (numOverlapsEnd1 == 0) && (numOverlapsEnd2 > 0) ) {
			for (vector<BED>::iterator q = qualityHits2.begin(); q != qualityHits2.end(); ++q) {
				_bedA->reportBedPETab(a);
				_bedB->reportBedNewLine(*q);
			}
		}
	}
	else if (type == "xor") {
		if ( (numOverlapsEnd1 > 0) && (numOverlapsEnd2 == 0) ) {
			for (vector<BED>::iterator q = qualityHits1.begin(); q != qualityHits1.end(); ++q) {
				_bedA->reportBedPETab(a);
				_bedB->reportBedNewLine(*q);
			}
		}
		else if ( (numOverlapsEnd1 == 0) && (numOverlapsEnd2 > 0) ) {
			for (vector<BED>::iterator q = qualityHits2.begin(); q != qualityHits2.end(); ++q) {
				_bedA->reportBedPETab(a);
				_bedB->reportBedNewLine(*q);
			}
		}
	}
	else if (type == "both") {
		if ( (numOverlapsEnd1 > 0) && (numOverlapsEnd2 > 0) ) {
			for (vector<BED>::iterator q = qualityHits1.begin(); q != qualityHits1.end(); ++q) {
				_bedA->reportBedPETab(a);
				_bedB->reportBedNewLine(*q);
			}
			for (vector<BED>::iterator q = qualityHits2.begin(); q != qualityHits2.end(); ++q) {
				_bedA->reportBedPETab(a);
				_bedB->reportBedNewLine(*q);
			}
		}
	}
}


bool BedIntersectPE::FindOneOrMoreOverlaps(const BEDPE &a, const string &type) {

	// flags for the existence of hits on each end of BEDPE
	// that exceed the requested overlap fraction
	bool end1Found = false;
	bool end2Found = false;

	// Look for overlaps in end 1 assuming we have an aligned chromosome.
	if (a.chrom1 != ".") {
		end1Found = _bedB->FindOneOrMoreOverlapsPerBin(a.chrom1, a.start1, a.end1, a.strand1, 
			_forceStrand, _overlapFraction);

		// can we bail out without checking end2?
		if ((type == "either") && (end1Found == true)) return true;
		else if ((type == "neither") && (end1Found == true)) return false;
		else if ((type == "notboth") && (end1Found == false)) return true;
		else if ((type == "both") && (end1Found == false)) return false;
	}
		
	// Now look for overlaps in end 2 assuming we have an aligned chromosome.
	if (a.chrom2 != ".") {
		end2Found = _bedB->FindOneOrMoreOverlapsPerBin(a.chrom2, a.start2, a.end2, a.strand2, 
			_forceStrand, _overlapFraction);
			
		if ((type == "either") && (end2Found == true)) return true;
		else if ((type == "neither") && (end2Found == true)) return false;
		else if ((type == "notboth") && (end2Found == false)) return true;
		else if ((type == "both") && (end2Found == false)) return false;		
	}
	
	// Now report the hits depending on what the user has requested.
	if (type == "notboth") {
		if ( (end1Found == false) || (end2Found == false) ) return true;
		else return false;
	}
	else if (type == "either") {
		if ( (end1Found == false) && (end2Found == false) ) return false;
	}
	else if (type == "neither") {
		if ( (end1Found == false) && (end2Found == false) ) return true;
		else return false;		
	}
	else if (type == "xor") {
		if ( (end1Found == true) && (end2Found == false) ) return true;
		else if ( (end1Found == false) && (end2Found == true) ) return true;
		else return false;
	}
	else if (type == "both") {
		if ( (end1Found == true) && (end2Found == true) ) return true;
		return false;
	}
	return false;
}


void BedIntersectPE::FindSpanningOverlaps(const BEDPE &a, vector<BED> &hits, const string &type) {

	// count of hits on _between_ end of BEDPE
	// that exceed the requested overlap fraction
	int numOverlaps = 0;
	int spanStart = 0;
	int spanEnd = 0;
	int spanLength = 0;
	
	if ((type == "ispan") || (type == "notispan")) {
		spanStart = a.end1;
		spanEnd = a.start2;
		if (a.end1 > a.start2) {
			spanStart = a.end2;
			spanEnd = a.start1;
		}
	}
	else if ((type == "ospan") || (type == "notospan")) {
		spanStart = a.start1;
		spanEnd = a.end2;		
		if (a.start1 > a.start2) {
			spanStart = a.start2;
			spanEnd = a.end1;
		}
	}
	spanLength = spanEnd - spanStart;

	// get the hits for the span
	_bedB->FindOverlapsPerBin(a.chrom1, spanStart, spanEnd, a.strand1, hits, _forceStrand);
	
	vector<BED>::const_iterator h = hits.begin();
	vector<BED>::const_iterator hitsEnd = hits.end();
	for (; h != hitsEnd; ++h) {
	
		int s = max(spanStart, h->start);
		int e = min(spanEnd, h->end);
		int overlapBases = (e - s);						// the number of overlapping bases b/w a and b
		int spanLength = (spanEnd - spanStart);		// the length of a in b.p.
		
		// is there enough overlap relative to the user's request? (default ~ 1bp)
		if ( ( (float) overlapBases / (float) spanLength ) >= _overlapFraction ) {
			numOverlaps++;
			if ((type == "ispan") || (type == "ospan")) {
				_bedA->reportBedPETab(a);
				_bedB->reportBedNewLine(*h);
			}
		}
	}
	
	if ( ( (type == "notispan") || (type == "notospan") ) && numOverlaps == 0 ) {
		_bedA->reportBedPENewLine(a);
	}
}


bool BedIntersectPE::FindOneOrMoreSpanningOverlaps(const BEDPE &a, const string &type) {

	int spanStart = 0;
	int spanEnd = 0;
	int spanLength = 0;
	bool overlapFound;
	
	if ((type == "ispan") || (type == "notispan")) {
		spanStart = a.end1;
		spanEnd = a.start2;
		if (a.end1 > a.start2) {
			spanStart = a.end2;
			spanEnd = a.start1;
		}
	}
	else if ((type == "ospan") || (type == "notospan")) {
		spanStart = a.start1;
		spanEnd = a.end2;		
		if (a.start1 > a.start2) {
			spanStart = a.start2;
			spanEnd = a.end1;
		}
	}
	spanLength = spanEnd - spanStart;

	overlapFound = _bedB->FindOneOrMoreOverlapsPerBin(a.chrom1, spanStart, spanEnd, a.strand1, 
		_forceStrand, _overlapFraction);

	return overlapFound;
}

 

void BedIntersectPE::IntersectBedPE(istream &bedInput) {

	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	_bedB->loadBedFileIntoMap();

	string bedLine;                                                                                                                    
	int lineNum = 0;					// current input line number
	vector<BED> hits, hits1, hits2;					// vector of potential hits
	vector<string> bedFields;			// vector for a BED entry
	
	// reserve some space
	hits.reserve(100);
	hits1.reserve(100);
	hits2.reserve(100);
	bedFields.reserve(12);	
		
	// process each entry in A
	while (getline(bedInput, bedLine)) {

		lineNum++;
		Tokenize(bedLine,bedFields);
		BEDPE a;
		
		// find the overlaps with B if it's a valid BED entry. 
		if (_bedA->parseBedPELine(a, bedFields, lineNum)) {
			if ( (_searchType == "ispan") || (_searchType == "ospan") ||
			 	 (_searchType == "notispan") || (_searchType == "notospan") ) {
				if (a.chrom1 == a.chrom2) {
					FindSpanningOverlaps(a, hits, _searchType);
					hits.clear();
				}
			}
			else {
				FindOverlaps(a, hits1, hits2, _searchType);
				hits1.clear();
				hits2.clear();
			}
		}
		// reset for the next input line
		bedFields.clear();
	}
}
// END IntersectPE


void BedIntersectPE::IntersectBamPE(string bamFile) {
	
	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	_bedB->loadBedFileIntoMap();
	
	// open the BAM file
	BamReader reader;
	BamWriter writer;
	reader.Open(bamFile);

	// get header & reference information
	string header = reader.GetHeaderText();
	RefVector refs = reader.GetReferenceData();

	// open a BAM output to stdout if we are writing BAM
	if (_bamOutput == true) {
		// open our BAM writer
		writer.Open("stdout", header, refs);
	}

	vector<BED> hits, hits1, hits2;		// vector of potential hits
	// reserve some space
	hits.reserve(1000);
	hits1.reserve(1000);
	hits2.reserve(1000);
		
	_bedA->bedType = 10;					// it's a full BEDPE given it's BAM
	BamAlignment bam;	
	bool overlapsFound;
	
	
	// get each set of alignments for each pair.
	while (reader.GetNextAlignment(bam)) {

		// endure that the BAM file is paired.
		if ( ! bam.IsPaired() ) {
			cerr << "Encountered an unpaired alignment.  Are you sure this is a paired BAM file?  Exiting." << endl;
			exit(1);
		}
		else if ((_searchType == "ispan") || (_searchType == "ospan")) {			
			// only look for ispan and ospan when both ends are mapped.
			if (bam.IsMapped() && bam.IsMateMapped()) {
				// only do an inspan or outspan check if the alignment is intrachromosomal
				if (bam.RefID == bam.MateRefID) {
					if (_bamOutput == true) {	// BAM output
						BEDPE a;  ConvertBamToBedPE(bam, refs, a);
						
						// look for overlaps, and write to BAM if >=1 were found	
						overlapsFound = FindOneOrMoreSpanningOverlaps(a, _searchType);
						if (overlapsFound == true) writer.SaveAlignment(bam);
					}
					else if ( IsCorrectMappingForBEDPE(bam) ) {	// BEDPE output
						BEDPE a;
						ConvertBamToBedPE(bam, refs, a);
						FindSpanningOverlaps(a, hits, _searchType);
					}
				}
			}		
		}
		else if ((_searchType == "notispan") || (_searchType == "notospan")) {
			// only look for notispan and notospan when both ends are mapped.
			if (bam.IsMapped() && bam.IsMateMapped()) {
				// only do an inspan or outspan if the alignment is intrachromosomal
				if (bam.RefID == bam.MateRefID) {
					if (_bamOutput == true) {	// BAM output
						BEDPE a;  ConvertBamToBedPE(bam, refs, a);
						
						// write to BAM if there were no overlaps
						overlapsFound = FindOneOrMoreSpanningOverlaps(a, _searchType);
						if (overlapsFound == false) writer.SaveAlignment(bam);
					}
					else if ( IsCorrectMappingForBEDPE(bam) ) {	// BEDPE output
						BEDPE a;
						ConvertBamToBedPE(bam, refs, a);
						FindSpanningOverlaps(a, hits, _searchType);
					}
				}
				// if inter-chromosomal or orphaned, we know it's not ispan and not ospan
				else if (_bamOutput == true) writer.SaveAlignment(bam);
			}
			// if both ends aren't mapped, we know that it's notispan and not ospan
			else if (_bamOutput == true) writer.SaveAlignment(bam);
		}
		else if ( (_searchType == "either") || (_searchType == "xor") || 
				  (_searchType == "both") || (_searchType == "notboth") ||
				  (_searchType == "neither") ) {
					
			if (_bamOutput == true) {	// BAM output
				BEDPE a;  ConvertBamToBedPE(bam, refs, a);

				// write to BAM if correct hits found
				overlapsFound = FindOneOrMoreOverlaps(a, _searchType);
				if (overlapsFound == true) writer.SaveAlignment(bam);
			}
			else if ( IsCorrectMappingForBEDPE(bam) ) {	// BEDPE output
				BEDPE a;
				ConvertBamToBedPE(bam, refs, a);
				FindOverlaps(a, hits1, hits2, _searchType);
				hits1.clear();
				hits2.clear();
			}
		}
	}
	reader.Close();
	if (_bamOutput == true) {
		writer.Close();
	}
}


void BedIntersectPE::DetermineBedPEInput() {
	
	if (_bedA->bedFile != "stdin") {   // process a file
		if (_bamInput == false) { // BEDPE
			ifstream beds(_bedA->bedFile.c_str(), ios::in);
			if ( !beds ) {
				cerr << "Error: The requested bed file (" << _bedA->bedFile << ") could not be opened. Exiting!" << endl;
				exit (1);
			}
			IntersectBedPE(beds);
		}
		else {	// bam
			IntersectBamPE(_bedA->bedFile);
		}
	}
	else {   // process stdin
		if (_bamInput == false) {	// BEDPE					
			IntersectBedPE(cin);
		}
		else {
			IntersectBamPE("stdin");
		}				
	}
}


