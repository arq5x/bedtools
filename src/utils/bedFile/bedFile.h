/*****************************************************************************
  bedFile.h

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licensed under the GNU General Public License 2.0 license.
******************************************************************************/
#ifndef BEDFILE_H
#define BEDFILE_H

// "local" includes
#include "gzstream.h"
#include "lineFileUtilities.h"
#include "fileType.h"

// standard includes
#include <vector>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <limits.h>
#include <stdint.h>
#include <cstdio>
//#include <tr1/unordered_map>  // Experimental.
using namespace std;


//*************************************************
// Data type tydedef
//*************************************************
typedef uint32_t CHRPOS;
typedef uint16_t BINLEVEL;
typedef uint32_t BIN;
typedef uint16_t USHORT;
typedef uint32_t UINT;

//*************************************************
// Genome binning constants
//*************************************************

const BIN      _numBins   = 37450;
const BINLEVEL _binLevels = 7;

// bins range in size from 16kb to 512Mb 
// Bin  0          spans 512Mbp,   # Level 1
// Bins 1-8        span 64Mbp,     # Level 2
// Bins 9-72       span 8Mbp,      # Level 3
// Bins 73-584     span 1Mbp       # Level 4
// Bins 585-4680   span 128Kbp     # Level 5
// Bins 4681-37449 span 16Kbp      # Level 6
const BIN _binOffsetsExtended[] = {32678+4096+512+64+8+1, 4096+512+64+8+1, 512+64+8+1, 64+8+1, 8+1, 1, 0};
//const BIN _binOffsetsExtended[] = {4096+512+64+8+1, 4096+512+64+8+1, 512+64+8+1, 64+8+1, 8+1, 1, 0};
	
const USHORT _binFirstShift = 14;	    /* How much to shift to get to finest bin. */
const USHORT _binNextShift  = 3;		/* How much to shift to get to next larger bin. */


//*************************************************
// Common data structures
//*************************************************

struct DEPTH {
	UINT starts;
	UINT ends;
};


/*
	Structure for regular BED records
*/
struct BED {

	// Regular BED fields
	string chrom;
	CHRPOS start;
	CHRPOS end; 
	string name;
	string score;
	string strand;

	// Add'l fields for BED12 and/or custom BED annotations
	vector<string> otherFields;

	// experimental fields for the FJOIN approach.
	bool   added;
	bool   finished;
	// list of hits from another file.
    vector<BED> overlaps;
	
public:
    // constructors

    // Null
    BED() 
    : chrom(""), 
      start(0),
      end(0),
      name(""),
      score(""),
      strand(""),
      otherFields(),
      added(false),
	  finished(false),
      overlaps()
    {}
        
    // BED3
    BED(string chrom, CHRPOS start, CHRPOS end) 
    : chrom(chrom), 
      start(start),
      end(end)
    {}

    // BED4
    BED(string chrom, CHRPOS start, CHRPOS end, string strand) 
    : chrom(chrom), 
      start(start),
      end(end),
      strand(strand)
    {}

    // BED6
    BED(string chrom, CHRPOS start, CHRPOS end, string name, 
        string score, string strand) 
    : chrom(chrom), 
      start(start),
      end(end),
      name(name),
      score(score),
      strand(strand)
    {}
    
    // BEDALL
    BED(string chrom, CHRPOS start, CHRPOS end, string name, 
        string score, string strand, vector<string> otherFields) 
    : chrom(chrom), 
      start(start),
      end(end),
      name(name),
      score(score),
      strand(strand),
      otherFields(otherFields)
    {}
    
}; // BED


/*
	Structure for each end of a paired BED record
	mate points to the other end.
*/
struct MATE {
    BED bed;
    int lineNum;
    MATE *mate;
};


/*
	Structure for regular BED COVERAGE records
*/
struct BEDCOV {

	// Regular BED fields
	CHRPOS start;
	CHRPOS end;
	
	string chrom;
	string name;
	string score;
	string strand;

	// Add'l fields for BED12 and/or custom BED annotations	
	vector<string> otherFields;

    // Additional fields specific to computing coverage
    map<unsigned int, DEPTH> depthMap;
    unsigned int count;
    CHRPOS minOverlapStart;
};


/*
	Structure for BED COVERAGE records having lists of
	multiple coverages
*/
struct BEDCOVLIST {

	// Regular BED fields
	CHRPOS start;
	CHRPOS end;
	
	string chrom;
	string name;
	string score;
	string strand;

	// Add'l fields for BED12 and/or custom BED annotations	
	vector<string> otherFields;

    // Additional fields specific to computing coverage
    vector< map<unsigned int, DEPTH> > depthMapList;
    vector<unsigned int> counts;
    vector<CHRPOS> minOverlapStarts;
};


// enum to flag the state of a given line in a BED file.
enum BedLineStatus
{ 
    BED_INVALID = -1,
    BED_HEADER  = 0,
    BED_BLANK   = 1,
    BED_VALID   = 2
};

// enum to indicate the type of file we are dealing with
enum FileType
{ 
    BED_FILETYPE,
    GFF_FILETYPE,
    VCF_FILETYPE
};

//*************************************************
// Data structure typedefs
//*************************************************
typedef vector<BED>    bedVector;
typedef vector<BEDCOV> bedCovVector;
typedef vector<MATE> mateVector;
typedef vector<BEDCOVLIST> bedCovListVector;

typedef map<BIN, bedVector,    std::less<BIN> > binsToBeds;
typedef map<BIN, bedCovVector, std::less<BIN> > binsToBedCovs;
typedef map<BIN, mateVector, std::less<BIN> > binsToMates;
typedef map<BIN, bedCovListVector, std::less<BIN> > binsToBedCovLists;

typedef map<string, binsToBeds, std::less<string> >    masterBedMap;
typedef map<string, binsToBedCovs, std::less<string> > masterBedCovMap;
typedef map<string, binsToMates, std::less<string> > masterMateMap;
typedef map<string, binsToBedCovLists, std::less<string> > masterBedCovListMap;
typedef map<string, bedVector, std::less<string> >     masterBedMapNoBin;


// EXPERIMENTAL - wait for TR1
// typedef vector<BED>    bedVector;
// typedef vector<BEDCOV> bedCovVector;
// 
// typedef tr1::unordered_map<BIN, bedVector> binsToBeds;
// typedef tr1::unordered_map<BIN, bedCovVector> binsToBedCovs;
// 
// typedef tr1::unordered_map<string, binsToBeds>    masterBedMap;
// typedef tr1::unordered_map<string, binsToBedCovs> masterBedCovMap;
// typedef tr1::unordered_map<string, bedVector>     masterBedMapNoBin;



// return the genome "bin" for a feature with this start and end
inline
BIN getBin(CHRPOS start, CHRPOS end) {
    --end;
    start >>= _binFirstShift;
    end   >>= _binFirstShift;
    
    for (register short i = 0; i < _binLevels; ++i) {
        if (start == end) return _binOffsetsExtended[i] + start;
        start >>= _binNextShift;
        end   >>= _binNextShift;
    }
    cerr << "start " << start << ", end " << end << " out of range in findBin (max is 512M)" << endl;
    return 0;
}

/****************************************************
// isInteger(s): Tests if string s is a valid integer
*****************************************************/
inline bool isInteger(const std::string& s) {
    int len = s.length();
    for (int i = 0; i < len; i++) {
        if (!std::isdigit(s[i])) return false;
    }    
    return true;
}


// return the amount of overlap between two features.  Negative if none and the the 
// number of negative bases is the distance between the two.
inline 
int overlaps(CHRPOS aS, CHRPOS aE, CHRPOS bS, CHRPOS bE) {
	return min(aE, bE) - max(aS, bS);
}


// Ancillary functions
void splitBedIntoBlocks(const BED &bed, int lineNum, bedVector &bedBlocks);


// BED Sorting Methods 
bool sortByChrom(const BED &a, const BED &b);	
bool sortByStart(const BED &a, const BED &b);
bool sortBySizeAsc(const BED &a, const BED &b);
bool sortBySizeDesc(const BED &a, const BED &b);
bool sortByScoreAsc(const BED &a, const BED &b);
bool sortByScoreDesc(const BED &a, const BED &b);
bool byChromThenStart(BED const &a, BED const &b);



//************************************************
// BedFile Class methods and elements
//************************************************
class BedFile {

public:

	// Constructor 
	BedFile(string &);

	// Destructor
	~BedFile(void);
	
	// Open a BED file for reading (creates an istream pointer)
	void Open(void);
	
	// Close an opened BED file.
	void Close(void);
	
	// Get the next BED entry in an opened BED file.
	BedLineStatus GetNextBed (BED &bed, int &lineNum);

	// load a BED file into a map keyed by chrom, then bin. value is vector of BEDs
	void loadBedFileIntoMap();

	// load a BED file into a map keyed by chrom, then bin. value is vector of BEDCOVs
	void loadBedCovFileIntoMap();
	
	// load a BED file into a map keyed by chrom, then bin. value is vector of BEDCOVLISTs
	void loadBedCovListFileIntoMap();

	// load a BED file into a map keyed by chrom. value is vector of BEDs
	void loadBedFileIntoMapNoBin();	

	// Given a chrom, start, end and strand for a single feature,
	// search for all overlapping features in another BED file.
	// Searches through each relevant genome bin on the same chromosome
	// as the single feature. Note: Adapted from kent source "binKeeperFind"
	void FindOverlapsPerBin(string chrom, CHRPOS start, CHRPOS end, string strand, vector<BED> &hits, bool forceStrand);

	// return true if at least one overlap was found.  otherwise, return false.
	bool FindOneOrMoreOverlapsPerBin(string chrom, CHRPOS start, CHRPOS end, string strand, 
										bool forceStrand, float overlapFraction = 0.0);

	// return true if at least one __reciprocal__ overlap was found.  otherwise, return false.
	bool FindOneOrMoreReciprocalOverlapsPerBin(string chrom, CHRPOS start, CHRPOS end, string strand, 
													bool forceStrand, float overlapFraction = 0.0);
	
	// Given a chrom, start, end and strand for a single feature,
	// increment a the number of hits for each feature in B file
	// that the feature overlaps
	void countHits(const BED &a, bool forceStrand);
	
	// same as above, but has special logic that processes a set of
	// BED "blocks" from a single entry so as to avoid over-counting 
	// each "block" of a single BAM/BED12 as distinct coverage.  That is,
	// if one read has four block, we only want to count the coverage as
	// coming from one read, not four.
    void countSplitHits(const vector<BED> &bedBlock, bool forceStrand);

	// Given a chrom, start, end and strand for a single feature,
	// increment a the number of hits for each feature in B file
	// that the feature overlaps	
    void countListHits(const BED &a, int index, bool forceStrand);
	
	// the bedfile with which this instance is associated
	string bedFile;
	unsigned int bedType;  // 3-6, 12 for BED
						   // 9 for GFF
	
	// Main data structires used by BEDTools
    masterBedCovMap      bedCovMap;
    masterBedCovListMap  bedCovListMap;
	masterBedMap         bedMap;
	masterBedMapNoBin    bedMapNoBin;
						
private:
	
	// data
	bool _isGff;
	bool _isVcf;
    bool _typeIsKnown;        // do we know the type?   (i.e., BED, GFF, VCF)
    FileType   _fileType;     // what is the file type? (BED? GFF? VCF?)    
	istream   *_bedStream;

    void setGff (bool isGff);
    void setVcf (bool isVcf);
    void setFileType (FileType type);
    void setBedType (int colNums);
    
    /******************************************************
    Private definitions to circumvent linker issues with
    templated member functions.
    *******************************************************/

    /*
        parseLine: converts a lineVector into either BED or BEDCOV (templated, hence in header to avoid linker issues.)
    */
    template <typename T>
    inline BedLineStatus parseLine (T &bed, const vector<string> &lineVector, int &lineNum) {

    	//char *p2End, *p3End, *p4End, *p5End;
    	//long l2, l3, l4, l5;
        unsigned int numFields = lineVector.size();
    	
    	// bail out if we have a blank line
    	if (numFields == 0) { 
    		return BED_BLANK;
		}

    	if ((lineVector[0].find("track") == string::npos) && (lineVector[0].find("browser") == string::npos) && (lineVector[0].find("#") == string::npos) ) {

    		if (numFields >= 3) {
    		    // line parsing for all lines after the first non-header line    		
                if (_typeIsKnown == true) {
                    switch(_fileType) {
                        case BED_FILETYPE:
                            if (parseBedLine(bed, lineVector, lineNum, numFields) == true) return BED_VALID;
                        case VCF_FILETYPE:
                            if (parseVcfLine(bed, lineVector, lineNum, numFields) == true) return BED_VALID;
                        case GFF_FILETYPE:
                            if (parseGffLine(bed, lineVector, lineNum, numFields) == true) return BED_VALID;
                        default:
                            printf("ERROR: file type encountered. Exiting\n");
                            exit(1);
                    }
                }
                // line parsing for first non-header line: figure out file contents 
                else {
        			// it's BED format if columns 2 and 3 are integers
        			if (isInteger(lineVector[1]) && isInteger(lineVector[2])) {
        				setGff(false);    
        				setFileType(BED_FILETYPE);
                        setBedType(numFields);       // we now expect numFields columns in each line
        				if (parseBedLine(bed, lineVector, lineNum, numFields) == true) return BED_VALID;
        			}
        			// it's VCF, assuming the second column is numeric and there are at least 8 fields.
        			else if (isInteger(lineVector[1]) && numFields >= 8) {    
                        setGff(false);
                        setVcf(true);
                        setFileType(VCF_FILETYPE);
                        setBedType(numFields);       // we now expect numFields columns in each line
                        if (parseVcfLine(bed, lineVector, lineNum, numFields) == true) return BED_VALID;
        			}
        			// it's GFF, assuming columns columns 4 and 5 are numeric and we have 9 fields total.
        			else if ((numFields == 9) && isInteger(lineVector[3]) && isInteger(lineVector[4])) {
    					setGff(true);
    					setFileType(GFF_FILETYPE);
    					setBedType(numFields);       // we now expect numFields columns in each line
    					if (parseGffLine(bed, lineVector, lineNum, numFields) == true) return BED_VALID;
        			}
        			else {
        				cerr << "Unexpected file format.  Please use tab-delimited BED, GFF, or VCF. " << 
        				        "Perhaps you have non-integer starts or ends at line " << lineNum << "?" << endl;
        				exit(1);
        			}
        		}
    		}
    		else {
    			cerr << "It looks as though you have less than 3 columns at line: " << lineNum << ".  Are you sure your files are tab-delimited?" << endl;
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


    /*
        parseBedLine: converts a lineVector into either BED or BEDCOV (templated, hence in header to avoid linker issues.)
    */
    template <typename T>
    inline bool parseBedLine (T &bed, const vector<string> &lineVector, int lineNum, unsigned int numFields) {

        // process as long as the number of fields in this 
        // line matches what we expect for this file.
        if (numFields == this->bedType) {        
            bed.chrom = lineVector[0];
            bed.start = atoi(lineVector[1].c_str());
            bed.end = atoi(lineVector[2].c_str());
            
            if (this->bedType == 4) {
                bed.name = lineVector[3];
            }
            else if (this->bedType == 5) {
                bed.name = lineVector[3];
                bed.score = lineVector[4];
            }
            else if (this->bedType == 6) {
                bed.name = lineVector[3];
                bed.score = lineVector[4];
                bed.strand = lineVector[5];
            }
            else if (this->bedType > 6) {
                bed.name = lineVector[3];
                bed.score = lineVector[4];
                bed.strand = lineVector[5];         
                for (unsigned int i = 6; i < lineVector.size(); ++i) {
                    bed.otherFields.push_back(lineVector[i]); 
                }
            }
            else if (this->bedType != 3) {
                cerr << "Error: unexpected number of fields at line: " << lineNum 
                     << ".  Verify that your files are TAB-delimited.  Exiting..." << endl; 
                exit(1);
            }
            
            // sanity checks.
            if ((bed.start <= bed.end) && (bed.start >= 0) && (bed.end >= 0)) {
                return true;
            }
            else if (bed.start > bed.end) {
                cerr << "Error: malformed BED entry at line " << lineNum << ". Start was greater than end. Exiting." << endl; 
                exit(1);
            }
            else if ( (bed.start < 0) || (bed.end < 0) ) {
                cerr << "Error: malformed BED entry at line " << lineNum << ". Coordinate detected that is < 0. Exiting." << endl;
                exit(1);
            }
        }
    	else if (numFields == 1) {
    		cerr << "Only one BED field detected: " << lineNum << ".  Verify that your files are TAB-delimited.  Exiting..." << endl;
    		exit(1);		
    	}
    	else if ((numFields != this->bedType) && (numFields != 0)) {
    		cerr << "Differing number of BED fields encountered at line: " << lineNum << ".  Exiting..." << endl;
    		exit(1);
    	}
    	else if ((numFields < 3) && (numFields != 0)) {
    		cerr << "TAB delimited BED file with at least 3 fields (chrom, start, end) is required at line: "<< lineNum << ".  Exiting..." << endl;
    		exit(1);
    	}
    	return false;
    }


    /*
        parseVcfLine: converts a lineVector into either BED or BEDCOV (templated, hence in header to avoid linker issues.)
    */
    template <typename T>
    inline bool parseVcfLine (T &bed, const vector<string> &lineVector, int lineNum, unsigned int numFields) {
    	if (numFields == this->bedType) { 
    		bed.chrom  = lineVector[0];
    		bed.start  = atoi(lineVector[1].c_str()) - 1;  // VCF is one-based
            bed.end    = bed.start + lineVector[3].size(); // VCF 4.0 stores the size of the affected REF allele.
            bed.strand = "+";
            // construct the name from the ref and alt alleles.  
            // if it's an annotated variant, add the rsId as well.
            bed.name   = lineVector[3] + "/" + lineVector[4];
            if (lineVector[2] != ".") {
                bed.name += "_" + lineVector[2];
            }

    		if (this->bedType > 2) {		
    			for (unsigned int i = 2; i < numFields; ++i)
    				bed.otherFields.push_back(lineVector[i]); 
    		}

    		if ((bed.start <= bed.end) && (bed.start > 0) && (bed.end > 0)) {
                return true;
    		}
    		else if (bed.start > bed.end) {
    			cerr << "Error: malformed VCF entry at line " << lineNum << ". Start was greater than end. Exiting." << endl;
    			exit(1);
    		}
    		else if ( (bed.start < 0) || (bed.end < 0) ) {
    			cerr << "Error: malformed VCF entry at line " << lineNum << ". Coordinate detected that is < 0. Exiting." << endl;
    			exit(1);
    		}
    	}
    	else if (numFields == 1) {
    		cerr << "Only one VCF field detected: " << lineNum << ".  Verify that your files are TAB-delimited.  Exiting..." << endl;
    		exit(1);		
    	}
    	else if ((numFields != this->bedType) && (numFields != 0)) {
    		cerr << "Differing number of VCF fields encountered at line: " << lineNum << ".  Exiting..." << endl;
    		exit(1);
    	}
    	else if ((numFields < 2) && (numFields != 0)) {
    		cerr << "TAB delimited VCF file with at least 2 fields (chrom, pos) is required at line: "<< lineNum << ".  Exiting..." << endl;
            exit(1);
    	}
    	return false;
    }



    /*
        parseGffLine: converts a lineVector into either BED or BEDCOV (templated, hence in header to avoid linker issues.)
    */
    template <typename T>
    inline bool parseGffLine (T &bed, const vector<string> &lineVector, int lineNum, unsigned int numFields) {
    	if (numFields == this->bedType) { 
    		if (this->bedType == 9 && _isGff) {
    			bed.chrom = lineVector[0];
    			// substract 1 to force the start to be BED-style
    			bed.start  = atoi(lineVector[3].c_str()) - 1;
    			bed.end    = atoi(lineVector[4].c_str());
    			bed.name   = lineVector[2];
    			bed.score  = lineVector[5];
    			bed.strand = lineVector[6].c_str();
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
    	else if (numFields == 1) {
    		cerr << "Only one GFF field detected: " << lineNum << ".  Verify that your files are TAB-delimited.  Exiting..." << endl;
    		exit(1);		
    	}
    	else if ((numFields != this->bedType) && (numFields != 0)) {
    		cerr << "Differing number of GFF fields encountered at line: " << lineNum << ".  Exiting..." << endl;
    		exit(1);
    	}
    	else if ((numFields < 9) && (numFields != 0)) {
    		cerr << "TAB delimited GFF file with 9 fields is required at line: "<< lineNum << ".  Exiting..." << endl;
    		exit(1);
    	}
    	return false;
    }
    
	
public:
    
    /*
    	reportBedTab

    	Writes the _original_ BED entry with a TAB
    	at the end of the line.
    	Works for BED3 - BED6.
    */
    template <typename T>
    inline void reportBedTab(const T &bed) {
        // BED
    	if (_isGff == false && _isVcf == false) {
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
    	// VCF
    	else if (_isGff == false && _isVcf == true) {
    	    printf ("%s\t%d\t", bed.chrom.c_str(), bed.start+1);

			vector<string>::const_iterator othIt = bed.otherFields.begin(); 
			vector<string>::const_iterator othEnd = bed.otherFields.end(); 
			for ( ; othIt != othEnd; ++othIt) {
				printf("%s\t", othIt->c_str());
			}
    	}	
    	// GFF
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
    template <typename T>
    inline void reportBedNewLine(const T &bed) {
        //BED
    	if (_isGff == false && _isVcf == false) {
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
    			printf ("%s\t%d\t%d\t%s\t%s\t%s", bed.chrom.c_str(), bed.start, bed.end, bed.name.c_str(), 
    												bed.score.c_str(), bed.strand.c_str());

    			vector<string>::const_iterator othIt = bed.otherFields.begin(); 
    			vector<string>::const_iterator othEnd = bed.otherFields.end(); 
    			for ( ; othIt != othEnd; ++othIt) {
    				printf("\t%s", othIt->c_str());
    			}
    			printf("\n");
    		}
    	}
    	// VCF
    	else if (_isGff == false && _isVcf == true) {
    	    printf ("%s\t%d\t", bed.chrom.c_str(), bed.start+1);

			vector<string>::const_iterator othIt = bed.otherFields.begin(); 
			vector<string>::const_iterator othEnd = bed.otherFields.end(); 
			for ( ; othIt != othEnd; ++othIt) {
				printf("%s\t", othIt->c_str());
			}
			printf("\n");
    	}
    	//GFF
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
    template <typename T>
    inline void reportBedRangeTab(const T &bed, CHRPOS start, CHRPOS end) {
        // BED
    	if (_isGff == false && _isVcf == false) {
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
    	// VCF
    	else if (_isGff == false && _isVcf == true) {
    	    printf ("%s\t%d\t", bed.chrom.c_str(), bed.start+1);

			vector<string>::const_iterator othIt = bed.otherFields.begin(); 
			vector<string>::const_iterator othEnd = bed.otherFields.end(); 
			for ( ; othIt != othEnd; ++othIt) {
				printf("%s\t", othIt->c_str());
			}
    	}
    	// GFF
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
    template <typename T>
    inline void reportBedRangeNewLine(const T &bed, CHRPOS start, CHRPOS end) {
        // BED
    	if (_isGff == false && _isVcf == false) {
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
    			printf ("%s\t%d\t%d\t%s\t%s\t%s", bed.chrom.c_str(), start, end, bed.name.c_str(), 
    												bed.score.c_str(), bed.strand.c_str());

    			vector<string>::const_iterator othIt = bed.otherFields.begin(); 
    			vector<string>::const_iterator othEnd = bed.otherFields.end(); 
    			for ( ; othIt != othEnd; ++othIt) {
    				printf("\t%s", othIt->c_str());
    			}
    			printf("\n");
    		}
    	}
    	// VCF
    	else if (_isGff == false && _isVcf == true) {
    	    printf ("%s\t%d\t", bed.chrom.c_str(), bed.start+1);

			vector<string>::const_iterator othIt = bed.otherFields.begin(); 
			vector<string>::const_iterator othEnd = bed.otherFields.end(); 
			for ( ; othIt != othEnd; ++othIt) {
				printf("%s\t", othIt->c_str());
			}
            printf("\n");
    	}
    	// GFF
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
    void reportNullBedTab() {

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
    void reportNullBedNewLine() {

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
    			printf (".\t-1\t-1\t.\t-1\t.");
    			for (unsigned int i = 6; i < this->bedType; ++i) {
    				printf("\t.");
    			}
    			printf("\n");
    		}
    	}	
    	else if (this->bedType == 9) {
    		printf (".\t.\t.\t-1\t-1\t-1\t.\t.\t.\n");
    	}
    }

    
};

#endif /* BEDFILE_H */
