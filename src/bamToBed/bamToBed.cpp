/*****************************************************************************
  bamToBed.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "version.h"
#include "BamReader.h"
#include "BamAncillary.h"
#include "BamAux.h"
#include "bedFile.h"
using namespace BamTools;

#include <vector>
#include <algorithm>	// for swap()
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;


// define our program name
#define PROGRAM_NAME "bamToBed"

// define our parameter checking macro
#define PARAMETER_CHECK(param, paramLen, actualLen) (strncmp(argv[i], param, min(actualLen, paramLen))== 0) && (actualLen == paramLen)


// function declarations
void ShowHelp(void);

void ConvertBamToBed(const string &bamFile, const bool &useEditDistance, 
                     const bool &writeBed12, const bool &obeySplits, const string &color);
void ConvertBamToBedpe(const string &bamFile, const bool &useEditDistance);
					
void PrintBed(const BamAlignment &bam, const RefVector &refs, bool useEditDistance, bool obeySplits);
void PrintBed12(const BamAlignment &bam, const RefVector &refs, bool useEditDistance, string color = "255,0,0");
void PrintBedPE(const BamAlignment &bam1, const BamAlignment &bam2,
                const RefVector &refs, bool useEditDistance);

void ParseCigarBed12(const vector<CigarOp> &cigar, vector<int> &blockStarts, 
                     vector<int> &blockEnds, int &alignmentEnd);
bool IsCorrectMappingForBEDPE (const BamAlignment &bam);


int main(int argc, char* argv[]) {

	// our configuration variables
	bool showHelp = false;

	// input files
	string bamFile;
	string color = "255,0,0";
	
	bool haveBam         = false;
	bool haveColor       = false;	
	bool writeBedPE      = false;
	bool writeBed12      = false;	
	bool useEditDistance = false;
	bool obeySplits      = false;
		
	// check to see if we should print out some help
	if(argc <= 1) showHelp = true;

	for(int i = 1; i < argc; i++) {
		int parameterLength = (int)strlen(argv[i]);

		if((PARAMETER_CHECK("-h", 2, parameterLength)) || 
		(PARAMETER_CHECK("--help", 5, parameterLength))) {
			showHelp = true;
		}
	}

	if(showHelp) ShowHelp();

	// do some parsing (all of these parameters require 2 strings)
	for(int i = 1; i < argc; i++) {

		int parameterLength = (int)strlen(argv[i]);

		if(PARAMETER_CHECK("-i", 2, parameterLength)) {
			if ((i+1) < argc) {
				haveBam = true;
				bamFile = argv[i + 1];
				i++;
			}
		}
		else if(PARAMETER_CHECK("-bedpe", 6, parameterLength)) {
				writeBedPE = true;
		}
		else if(PARAMETER_CHECK("-bed12", 6, parameterLength)) {
				writeBed12 = true;
		}	
		else if(PARAMETER_CHECK("-split", 6, parameterLength)) {
				obeySplits = true;
		}	
		else if(PARAMETER_CHECK("-ed", 3, parameterLength)) {
				useEditDistance = true;
		}
		else if(PARAMETER_CHECK("-color", 6, parameterLength)) {
			if ((i+1) < argc) {
				haveColor = true;
				color = argv[i + 1];
				i++;
			}	
		}				
		else {
			cerr << endl << "*****ERROR: Unrecognized parameter: " << argv[i] << " *****" << endl << endl;
			showHelp = true;
		}		
	}

	// make sure we have an input files
	if (haveBam == false) {
		cerr << endl << "*****" << endl << "*****ERROR: Need -i (BAM) file. " << endl << "*****" << endl;
		showHelp = true;
	}
	if (haveColor == true && writeBed12 == false) {
		cerr << endl << "*****" << endl << "*****ERROR: Cannot use color without BED12. " << endl << "*****" << endl;
		showHelp = true;
	}
	if (useEditDistance == true && obeySplits == true) {
		cerr << endl << "*****" << endl << "*****ERROR: Cannot use -ed with -splits.  Edit distances cannot be computed for each \'chunk\'." << endl << "*****" << endl;
		showHelp = true;
	}
	
	// if there are no problems, let's convert BAM to BED or BEDPE
	if (!showHelp) {
		if (writeBedPE == false) 
			ConvertBamToBed(bamFile, useEditDistance, writeBed12, obeySplits, color);	// BED or "blocked BED"
		else
			ConvertBamToBedpe(bamFile, useEditDistance);                    // BEDPE
	}	
	else {
		ShowHelp();
	}
}


void ShowHelp(void) {

	cerr << endl << "Program: " << PROGRAM_NAME << " (v" << VERSION << ")" << endl;
	
	cerr << "Author:  Aaron Quinlan (aaronquinlan@gmail.com)" << endl;

	cerr << "Summary: Converts BAM alignments to BED6 or BEDPE format." << endl << endl;

	cerr << "Usage:   " << PROGRAM_NAME << " [OPTIONS] -i <bam> " << endl << endl;

	cerr << "Options: " << endl;
	
	cerr << "\t-bedpe\t"	<< "Write BEDPE format." << endl;
	cerr 					<< "\t\t- Requires BAM to be grouped or sorted by query." << endl << endl;
	
	cerr << "\t-bed12\t"	<< "Write \"blocked\" BED format (aka \"BED12\")." << endl << endl;
	cerr 					<< "\t\thttp://genome-test.cse.ucsc.edu/FAQ/FAQformat#format1" << endl << endl;

	cerr << "\t-split\t"	<< "Report \"split\" BAM alignments as separate BED entries." << endl << endl;
			
	cerr << "\t-ed\t"		<< "Use BAM edit distance (NM tag) for BED score." << endl;
	cerr 					<< "\t\t- Default for BED is to use mapping quality." << endl;
	cerr 					<< "\t\t- Default for BEDPE is to use the minimum of" << endl;
	cerr 					<< "\t\t  the two mapping qualities for the pair." << endl;
	cerr 					<< "\t\t- When -ed is used with -bedpe, the total edit" << endl;
	cerr 					<< "\t\t  distance from the two mates is reported." << endl << endl;

	cerr << "\t-color\t"	<< "An R,G,B string for the color used with BED12 format." << endl;
	cerr 					<< "\t\tDefault is (255,0,0)." << endl;


	// end the program here
	exit(1);
}


void ConvertBamToBed(const string &bamFile, const bool &useEditDistance, 
                     const bool &writeBed12, const bool &obeySplits, const string &color) {
	// open the BAM file
	BamReader reader;
	reader.Open(bamFile);

	// get header & reference information
	string header = reader.GetHeaderText();
	RefVector refs = reader.GetReferenceData();
	
	// rip through the BAM file and convert each mapped entry to BED
	BamAlignment bam;
	while (reader.GetNextAlignment(bam)) {
		if (bam.IsMapped() == true) {
			if (writeBed12 == false)                 // BED
				PrintBed(bam, refs, useEditDistance, obeySplits);
			else                                     //"blocked" BED
				PrintBed12(bam, refs, useEditDistance, color);
		}
	}
	reader.Close();	
}


/*
  Assumptions:
     1.  The BAM file is grouped/sorted by query name, 
         not alignment position
*/
void ConvertBamToBedpe(const string &bamFile, const bool &useEditDistance) {
	// open the BAM file
	BamReader reader;
	reader.Open(bamFile);

	// get header & reference information
	string header = reader.GetHeaderText();
	RefVector refs = reader.GetReferenceData();
	
	// rip through the BAM file and convert each mapped entry to BEDPE
	BamAlignment bam1, bam2;
	while (reader.GetNextAlignment(bam1)) {
		// the alignment must be paired
		if (bam1.IsPaired() == true) {
			// grab the second alignment for the pair.
			reader.GetNextAlignment(bam2);
			
			// require that the alignments are from the same query
			if (bam1.Name == bam2.Name) {
				PrintBedPE(bam1, bam2, refs, useEditDistance);
			}
			else {
				cerr << "*****ERROR: -bedpe requires BAM to be sorted/grouped by query name. " << endl;
				exit(1);
			}
		}
	}
	reader.Close();
}


void ParseCigarBed12(const vector<CigarOp> &cigar, vector<int> &blockStarts, vector<int> &blockLengths, unsigned int &alignmentEnd) {

	int currPosition = 0;
	int blockLength	 = 0;
		
	//	Rip through the CIGAR ops and figure out if there is more 
	//	than one block for this alignment
	vector<CigarOp>::const_iterator cigItr = cigar.begin();
	vector<CigarOp>::const_iterator cigEnd = cigar.end();
	for (; cigItr != cigEnd; ++cigItr) {
		switch (cigItr->Type) {
			case ('M') : 
				blockLength  += cigItr->Length;
				currPosition += cigItr->Length;
			case ('I') : break;
		    case ('S') : break;
		    case ('D') : break;
				blockLength  += cigItr->Length;
				currPosition += cigItr->Length;
		    case ('P') : break;
			case ('N') :
				blockStarts.push_back(currPosition + cigItr->Length);
				blockLengths.push_back(blockLength);
				currPosition += cigItr->Length;
				blockLength = 0;
		    case ('H') : break; 					        // for 'H' - do nothing, move to next op
		    default    : 
				printf("ERROR: Invalid Cigar op type\n");   // shouldn't get here
				exit(1);
		}
	}
	// add the kast block and set the 
	// alignment end (i.e., relative to the start)
	blockLengths.push_back(blockLength);
	alignmentEnd = currPosition;
}


void PrintBed(const BamAlignment &bam,  const RefVector &refs, bool useEditDistance, bool obeySplits) {

	// set the strand
	string strand = "+"; 
	if (bam.IsReverseStrand() == true) strand = "-";

	// set the name of the feature based on the sequence
	string name = bam.Name;
	if (bam.IsFirstMate() == true)  name += "/1";
	if (bam.IsSecondMate() == true) name += "/2";

	// get the unpadded (parm = false) end position based on the CIGAR
	unsigned int alignmentEnd = bam.GetEndPosition(false);

    // report the entire BAM footprint as a single BED entry
    if (obeySplits == false) {
    	// report the alignment in BED6 format.
    	if (useEditDistance == false) {
    		printf("%s\t%d\t%d\t\%s\t%d\t%s\n", refs.at(bam.RefID).RefName.c_str(), bam.Position,
    									  alignmentEnd, name.c_str(), bam.MapQuality, strand.c_str());
    	}
    	else {
    		uint8_t editDistance;
    		if (bam.GetEditDistance(editDistance)) {
    			printf("%s\t%d\t%d\t\%s\t%u\t%s\n", refs.at(bam.RefID).RefName.c_str(), bam.Position,
    										  alignmentEnd, name.c_str(), editDistance, strand.c_str());
    		}
    		else {
    			cerr << "The edit distance tag (NM) was not found in the BAM file.  Please disable -ed.  Exiting\n";
    			exit(1);
    		}
    	}
	}
    // Report each chunk of the BAM alignment as a discrete BED entry
    // For example 10M100N10M would be reported as two seprate BED entries of length 10
	else {
        vector<BED> bedBlocks;
        // Load the alignment blocks in bam into the bedBlocks vector. 
        // Don't trigger a new block when a "D" (deletion) CIGAR op is found.
        getBamBlocks(bam, refs, bedBlocks, false);
        
        vector<BED>::const_iterator bedItr = bedBlocks.begin();
    	vector<BED>::const_iterator bedEnd = bedBlocks.end();
    	for (; bedItr != bedEnd; ++bedItr) {
	        printf("%s\t%d\t%d\t\%s\t%d\t%s\n", refs.at(bam.RefID).RefName.c_str(), bedItr->start,
    									  bedItr->end, name.c_str(), bam.MapQuality, strand.c_str());
	    }
	}
}


void PrintBed12(const BamAlignment &bam, const RefVector &refs, bool useEditDistance, string color) {

	// set the strand
	string strand = "+"; 
	if (bam.IsReverseStrand()) strand = "-";

	// set the name of the feature based on the sequence
	string name = bam.Name;
	if (bam.IsFirstMate()) name += "/1";
	if (bam.IsSecondMate()) name += "/2";
	
	// parse the CIGAR string and figure out the alignment blocks
	unsigned int alignmentEnd;
	vector<int> blockLengths;
	vector<int> blockStarts;
	blockStarts.push_back(0);
	   
	// extract the block starts and lengths from the CIGAR string
	ParseCigarBed12(bam.CigarData, blockStarts, blockLengths, alignmentEnd);
	alignmentEnd += bam.Position;
	
	// write BED6 portion
	if (useEditDistance == false) {
		printf("%s\t%d\t%d\t\%s\t%d\t%s\t", refs.at(bam.RefID).RefName.c_str(), bam.Position,
									  alignmentEnd, name.c_str(), bam.MapQuality, strand.c_str());
	}
	else {
		uint8_t editDistance;
		if (bam.GetEditDistance(editDistance)) {
			printf("%s\t%d\t%d\t\%s\t%u\t%s\t", refs.at(bam.RefID).RefName.c_str(), bam.Position,
										  alignmentEnd, name.c_str(), editDistance, strand.c_str());
		}
		else {
			cerr << "The edit distance tag (NM) was not found in the BAM file.  Please disable -ed.  Exiting\n";
			exit(1);
		}
	}
	
	// write the colors, etc.
	printf("%d\t%d\t%s\t%d\t", bam.Position, alignmentEnd, color.c_str(), (int) blockStarts.size());
	
	// now write the lengths portion
	unsigned int b;
	for (b = 0; b < blockLengths.size() - 1; ++b) {
		printf("%d,", blockLengths[b]);
	}
	printf("%d\t", blockLengths[b]);

	// now write the starts portion
	for (b = 0; b < blockStarts.size() - 1; ++b) {
		printf("%d,", blockStarts[b]);
	}
	printf("%d\n", blockStarts[b]);	
}


void PrintBedPE(const BamAlignment &bam1, const BamAlignment &bam2, const RefVector &refs, bool useEditDistance) {

	// initialize BEDPE variables
	string chrom1, chrom2, strand1, strand2;
	int start1, start2, end1, end2;
	uint8_t editDistance1, editDistance2;
	start1 = start2 = end1 = end2 = -1;
	chrom1 = chrom2 = strand1 = strand2 = ".";
	editDistance1 = editDistance2 = 0;
	uint16_t minMapQuality = 0;		
				
	// extract relevant info for end 1
	if (bam1.IsMapped()) {
		chrom1 = refs.at(bam1.RefID).RefName;
		start1 = bam1.Position;
		end1   = bam1.GetEndPosition(false);
		strand1 = "+";
		if (bam1.IsReverseStrand()) strand1 = "-";
		
		// extract the edit distance from the NM tag
		// if possible. otherwise, complain.
		if (useEditDistance == true && bam1.GetEditDistance(editDistance1) == false) {
			cerr << "The edit distance tag (NM) was not found in the BAM file.  Please disable -ed.  Exiting\n";
			exit(1);
		}
	}

	// extract relevant info for end 2
	if (bam2.IsMapped()) {
		chrom2 = refs.at(bam2.RefID).RefName;
		start2 = bam2.Position;
		end2   = bam2.GetEndPosition(false);
		strand2 = "+";
		if (bam2.IsReverseStrand()) strand2 = "-";
		
		// extract the edit distance from the NM tag
		// if possible. otherwise, complain.
		if (useEditDistance == true && bam2.GetEditDistance(editDistance2) == false) {
			cerr << "The edit distance tag (NM) was not found in the BAM file.  Please disable -ed.  Exiting\n";
			exit(1);
		}
	}
	
	// swap the ends if necessary
	if ( chrom1 > chrom2 || ((chrom1 == chrom2) && (start1 > start2)) ) {
		swap(chrom1, chrom2);
		swap(start1, start2);
		swap(end1, end2);
		swap(strand1, strand2);		
	}

	// report BEDPE using min mapQuality
	if (useEditDistance == false) {
		// compute the minimum mapping quality b/w the two ends of the pair.
		if (bam1.IsMapped() == true && bam2.IsMapped() == true) 
			minMapQuality = min(bam1.MapQuality, bam2.MapQuality); 
		   
		printf("%s\t%d\t%d\t\%s\t%d\t%d\t%s\t%d\t%s\t%s\n", 
				chrom1.c_str(), start1, end1, chrom2.c_str(), start2, end2, 
				bam1.Name.c_str(), minMapQuality, strand1.c_str(), strand2.c_str());
	}
	// report BEDPE using total edit distance
	else {	
		uint16_t totalEditDistance = 0;
		if (bam1.IsMapped() == true && bam2.IsMapped() == true)
			totalEditDistance = editDistance1 + editDistance2;
		else if (bam1.IsMapped() == true)
			totalEditDistance = editDistance1;
		else if (bam2.IsMapped() == true)
			totalEditDistance = editDistance2;

		printf("%s\t%d\t%d\t\%s\t%d\t%d\t%s\t%d\t%s\t%s\n", 
				chrom1.c_str(), start1, end1, chrom2.c_str(), start2, end2, 
				bam1.Name.c_str(), totalEditDistance, strand1.c_str(), strand2.c_str());			
	}
}


// deprecated.
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
