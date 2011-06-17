/*****************************************************************************
  windowBed.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0 license.
******************************************************************************/
#include "lineFileUtilities.h"
#include "windowBed.h"


/*
    Constructor
*/
BedWindow::BedWindow(string bedAFile, string bedBFile, int leftSlop, int rightSlop,
                     bool anyHit, bool noHit, bool writeCount, bool strandWindows,
                     bool matchOnStrand, bool bamInput, bool bamOutput, bool isUncompressedBam) {

    _bedAFile      = bedAFile;
    _bedBFile      = bedBFile;

    _leftSlop      = leftSlop;
    _rightSlop     = rightSlop;

    _anyHit              = anyHit;
    _noHit               = noHit;
    _writeCount          = writeCount;
    _strandWindows       = strandWindows;
    _matchOnStrand       = matchOnStrand;
    _bamInput            = bamInput;
    _bamOutput           = bamOutput;
    _isUncompressedBam   = isUncompressedBam;

    _bedA          = new BedFile(bedAFile);
    _bedB          = new BedFile(bedBFile);

    if (_bamInput == false)
        WindowIntersectBed();
    else
        WindowIntersectBam(_bedAFile);
}



/*
    Destructor
*/
BedWindow::~BedWindow(void) {
}



void BedWindow::FindWindowOverlaps(const BED &a, vector<BED> &hits) {

    /*
        Adjust the start and end of a based on the requested window
    */

    // update the current feature's start and end
    // according to the slop requested (slop = 0 by default)
    CHRPOS aFudgeStart = 0;
    CHRPOS aFudgeEnd;
    AddWindow(a, aFudgeStart, aFudgeEnd);

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
        int overlapBases = (e - s);             // the number of overlapping bases b/w a and b
        int aLength = (a.end - a.start);        // the length of a in b.p.

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
        _bedA->reportBedNewLine(a); }
    else if (_writeCount == true) {
        _bedA->reportBedTab(a); printf("\t%d\n", numOverlaps);
    }
    else if (_noHit == true && (numOverlaps == 0)) {
        _bedA->reportBedNewLine(a);
    }
}


bool BedWindow::FindOneOrMoreWindowOverlaps(const BED &a) {

    // update the current feature's start and end
    // according to the slop requested (slop = 0 by default)
    CHRPOS aFudgeStart = 0;
    CHRPOS aFudgeEnd;
    AddWindow(a, aFudgeStart, aFudgeEnd);

    bool overlapsFound = _bedB->FindOneOrMoreOverlapsPerBin(a.chrom, a.start, a.end, a.strand, _matchOnStrand);
    return overlapsFound;
}


void BedWindow::WindowIntersectBed() {

    // load the "B" bed file into a map so
    // that we can easily compare "A" to it for overlaps
    _bedB->loadBedFileIntoMap();

    BED a, nullBed;
    int lineNum = 0;                    // current input line number
    BedLineStatus bedStatus;
    vector<BED> hits;                   // vector of potential hits
    hits.reserve(100);

    _bedA->Open();
    while ((bedStatus = _bedA->GetNextBed(a, lineNum)) != BED_INVALID) {
        if (bedStatus == BED_VALID) {
            FindWindowOverlaps(a, hits);
            hits.clear();
            a = nullBed;
        }
    }
    _bedA->Close();
}


void BedWindow::WindowIntersectBam(string bamFile) {

    // load the "B" bed file into a map so
    // that we can easily compare "A" to it for overlaps
    _bedB->loadBedFileIntoMap();

    // open the BAM file
    BamReader reader;
    BamWriter writer;
    reader.Open(bamFile);

    // get header & reference information
    string bamHeader  = reader.GetHeaderText();
    RefVector refs    = reader.GetReferenceData();

    // open a BAM output to stdout if we are writing BAM
    if (_bamOutput == true) {
        // set compression mode
        BamWriter::CompressionMode compressionMode = BamWriter::Compressed;
        if ( _isUncompressedBam ) compressionMode = BamWriter::Uncompressed;
        writer.SetCompressionMode(compressionMode);
        // open our BAM writer
        writer.Open("stdout", bamHeader, refs);
    }

    vector<BED> hits;                   // vector of potential hits
    // reserve some space
    hits.reserve(100);

    _bedA->bedType = 6;
    BamAlignment bam;
    bool overlapsFound;
    // get each set of alignments for each pair.
    while (reader.GetNextAlignment(bam)) {

        if (bam.IsMapped()) {
            BED a;
            a.chrom = refs.at(bam.RefID).RefName;
            a.start = bam.Position;
            a.end   = bam.GetEndPosition(false, false);

            // build the name field from the BAM alignment.
            a.name = bam.Name;
            if (bam.IsFirstMate()) a.name += "/1";
            if (bam.IsSecondMate()) a.name += "/2";

            a.score  = ToString(bam.MapQuality);
            a.strand = "+"; if (bam.IsReverseStrand()) a.strand = "-";

            if (_bamOutput == true) {
                overlapsFound = FindOneOrMoreWindowOverlaps(a);
                if (overlapsFound == true) {
                    if (_noHit == false)
                        writer.SaveAlignment(bam);
                }
                else {
                    if (_noHit == true)
                        writer.SaveAlignment(bam);
                }
            }
            else {
                FindWindowOverlaps(a, hits);
                hits.clear();
            }
        }
        // BAM IsMapped() is false
        else if (_noHit == true) {
            writer.SaveAlignment(bam);
        }
    }

    // close the relevant BAM files.
    reader.Close();
    if (_bamOutput == true) {
        writer.Close();
    }
}


void BedWindow::AddWindow(const BED &a, CHRPOS &fudgeStart, CHRPOS &fudgeEnd) {
    // Does the user want to treat the windows based on strand?
    // If so,
    // if "+", then left is left and right is right
    // if "-", the left is right and right is left.
    if (_strandWindows) {
        if (a.strand == "+") {
            if ((int) (a.start - _leftSlop) > 0) 
                fudgeStart = a.start - _leftSlop;
            else fudgeStart = 0;
            fudgeEnd = a.end + _rightSlop;
        }
        else {
            if ((int) (a.start - _rightSlop) > 0) 
                fudgeStart = a.start - _rightSlop;
            else fudgeStart = 0;
            fudgeEnd = a.end + _leftSlop;
        }
    }
    // If not, add the windows irrespective of strand
    else {
        if ((int) (a.start - _leftSlop) > 0) 
            fudgeStart = a.start - _leftSlop;
        else fudgeStart = 0;
        fudgeEnd = a.end + _rightSlop;
    }
}

