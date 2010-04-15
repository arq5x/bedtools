/*****************************************************************************
  bamToBed.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "lineFileUtilities.h"
#include "bedFile.h"
#include "genomeFile.h"
#include "version.h"

#include "BamWriter.h"
#include "BamAux.h"
using namespace BamTools;

#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;


// define our program name
#define PROGRAM_NAME "bedToBam"

// define our parameter checking macro
#define PARAMETER_CHECK(param, paramLen, actualLen) (strncmp(argv[i], param, min(actualLen, paramLen))== 0) && (actualLen == paramLen)


// function declarations
void ShowHelp(void);
void DetermineBedInput(BedFile *bed, GenomeFile *genome, bool isBED12, int mapQual);
void ProcessBed(istream &bedInput, BedFile *bed, GenomeFile *genome, bool isBED12, int mapQual);
void ConvertBedToBam(const BED &bed, BamAlignment &bam, map<string, int> &chromToId, bool isBED12, int mapQual, int lineNum);
void MakeBamHeader(const string &genomeFile, RefVector &refs, string &header, map<string, int> &chromToInt);
int  reg2bin(int beg, int end);



int main(int argc, char* argv[]) {

	// our configuration variables
	bool showHelp = false;

	// input files
	string bedFile;
	string genomeFile;
	
	unsigned int mapQual = 255;
	
	bool haveBed         = false;
	bool haveGenome      = false;	
	bool haveMapQual     = false;
	bool isBED12         = false;

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
				haveBed = true;
				bedFile = argv[i + 1];
				i++;
			}
		}
		else if(PARAMETER_CHECK("-g", 2, parameterLength)) {
			if ((i+1) < argc) {
				haveGenome = true;
				genomeFile = argv[i + 1];
				i++;
			}
		}
		else if(PARAMETER_CHECK("-mapq", 5, parameterLength)) {
			haveMapQual = true;
			if ((i+1) < argc) {
				mapQual = atoi(argv[i + 1]);
				i++;
			}
		}	
		else if(PARAMETER_CHECK("-bed12", 6, parameterLength)) {
			isBED12 = true;
		}			
		else {
			cerr << endl << "*****ERROR: Unrecognized parameter: " << argv[i] << " *****" << endl << endl;
			showHelp = true;
		}		
	}

	// make sure we have an input files
	if (!haveBed ) {
		cerr << endl << "*****" << endl << "*****ERROR: Need -i (BED) file. " << endl << "*****" << endl;
		showHelp = true;
	}
	if (!haveGenome ) {
		cerr << endl << "*****" << endl << "*****ERROR: Need -g (genome) file. " << endl << "*****" << endl;
		showHelp = true;
	}
	if (mapQual < 0 || mapQual > 255) {
		cerr << endl << "*****" << endl << "*****ERROR: MAPQ must be in range [0,255]. " << endl << "*****" << endl;
		showHelp = true;
	}

	
	if (!showHelp) {
		BedFile *bed       = new BedFile(bedFile);
		GenomeFile *genome = new GenomeFile(genomeFile);
		
		DetermineBedInput(bed, genome, isBED12, mapQual);
	}	
	else {
		ShowHelp();
	}
}


void ShowHelp(void) {

	cerr << endl << "Program: " << PROGRAM_NAME << " (v" << VERSION << ")" << endl;
	
	cerr << "Author:  Aaron Quinlan (aaronquinlan@gmail.com)" << endl;

	cerr << "Summary: Converts BED records to BAM format." << endl << endl;

	cerr << "Usage:   " << PROGRAM_NAME << " [OPTIONS] -i <bed> -g <genome>" << endl << endl;

	cerr << "Options: " << endl;

	cerr << "\t-mapq\t"	<< "Set the mappinq quality for the BAM records." << endl;
	cerr 					<< "\t\t(INT) Default: 255" << endl << endl;

	cerr << "\t-bed12\t"	<< "The BED file is in BED12 format.  The BAM CIGAR" << endl;
	cerr 					<< "\t\tstring will reflect BED \"blocks\"." << endl << endl;


	cerr << "Notes: " << endl;
	cerr << "\t(1)  BED files must be at least BED4 (needs name field)." << endl << endl;


	// end the program here
	exit(1);
}


void DetermineBedInput(BedFile *bed, GenomeFile *genome, bool isBED12, int mapQual) {
	
	// dealing with a proper file
	if (bed->bedFile != "stdin") {   

		ifstream bedStream(bed->bedFile.c_str(), ios::in);
		if ( !bedStream ) {
			cerr << "Error: The requested bed file (" << bed->bedFile << ") could not be opened. Exiting!" << endl;
			exit (1);
		}
		ProcessBed(bedStream, bed, genome, isBED12, mapQual);
	}
	// reading from stdin
	else {  					
		ProcessBed(cin, bed, genome, isBED12, mapQual);
	}
}


void ProcessBed(istream &bedInput, BedFile *bed, GenomeFile *genome, bool isBED12, int mapQual) {

	// open the BAM file for writing.
	BamWriter writer;
	
	// build a BAM header from the genomeFile
	RefVector refs;
	string    bamHeader;
	map<string, int, std::less<string> > chromToId;
	MakeBamHeader(genome->getGenomeFileName(), refs, bamHeader, chromToId);
	
	// add the reference headers to the BAM file
	writer.Open("stdout", bamHeader, refs);


	string bedLine;                                                                                                                    
	int lineNum = 0;					// current input line number
	vector<string> bedFields;			// vector for a BED entry
	bedFields.reserve(12);	
		
	// process each entry in A
	while (getline(bedInput, bedLine)) {

		lineNum++;
		Tokenize(bedLine,bedFields);
		
		BED theBed;
		BamAlignment theBam;
		if (bed->parseLine(theBed, bedFields, lineNum)) {			
			if (bed->bedType >= 4) {
				ConvertBedToBam(theBed, theBam, chromToId, isBED12, mapQual, lineNum);
				writer.SaveAlignment(theBam);
			}
			else {
				cerr << "Error: BED entry without name found at line: " << lineNum << ".  Exiting!" << endl;
				exit (1);
			}
		}
		// reset for the next input line
		bedFields.clear();
	}
	writer.Close();
}


void ConvertBedToBam(const BED &bed, BamAlignment &bam, map<string, int, std::less<string> > &chromToId, 
                     bool isBED12, int mapQual, int lineNum) {
	
	bam.Name       = bed.name;
	bam.Position   = bed.start;
	bam.Bin        = reg2bin(bed.start, bed.end);
	
	// hard-code the sequence and qualities.
	int bedLength  = bed.end - bed.start;
	string query(bedLength, 'N');
	string quals(bedLength, 'H');
	bam.QueryBases = query;
	bam.Qualities  = quals;

	// chrom and map quality
	bam.RefID      = chromToId[bed.chrom];
	bam.MapQuality = mapQual;
	
	// set the BAM FLAG
	bam.AlignmentFlag = 0;
	if (bed.strand == "-")
		bam.SetIsReverseStrand(true);
	
	bam.MatePosition = -1;
	bam.InsertSize   = 0;
	bam.MateRefID    = -1;
	
	if (isBED12 == false) {
		CigarOp cOp;
		cOp.Type = 'M';
		cOp.Length = bedLength;
		bam.CigarData.push_back(cOp);
	}
	// precess each block.
	else {

		// extract the relevant BED fields to convert BED12 to BAM
		// namely: thickStart, thickEnd, blockCount, blockStarts, blockEnds
		// unsigned int thickStart = atoi(bed.otherFields[0].c_str());
		// unsigned int thickEnd   = atoi(bed.otherFields[1].c_str());
		unsigned int blockCount = atoi(bed.otherFields[3].c_str());

		vector<string> blockSizesString, blockStartsString;
		vector<int> blockSizes, blockStarts;
		Tokenize(bed.otherFields[4], blockSizesString, ",");
		Tokenize(bed.otherFields[5], blockStartsString, ",");
		
		for (unsigned int i = 0; i < blockCount; ++i) {
			blockStarts.push_back(atoi(blockStartsString[i].c_str()));
			blockSizes.push_back(atoi(blockSizesString[i].c_str()));
		}
		
		// make sure this is a well-formed BED12 entry.
		if ((blockSizes.size() != blockCount) || (blockSizes.size() != blockCount)) {
			cerr << "Error: Number of BED blocks does not match blockCount at line: " << lineNum << ".  Exiting!" << endl;
			exit (1);
		}
		else {
			// does the first block start after the bed.start?
			// if so, we need to do some "splicing"
			if (blockStarts[0] - bed.start > 0) {	
				CigarOp cOp;
				cOp.Length = blockStarts[0] - bed.start;
				cOp.Type = 'N';
				bam.CigarData.push_back(cOp);
			}
			// handle the "middle" blocks
			for (unsigned int i = 0; i < blockCount - 1; ++i) {
				CigarOp cOp;
				cOp.Length = blockSizes[i];
				cOp.Type = 'M';
				bam.CigarData.push_back(cOp);
			
				if (blockStarts[i+1] > (blockStarts[i] + blockSizes[i])) {
					CigarOp cOp;
					cOp.Length = (blockStarts[i+1] - (blockStarts[i] + blockSizes[i]));
					cOp.Type = 'N';
					bam.CigarData.push_back(cOp);
				}
			}
			// handle the last block.
			CigarOp cOp;
			cOp.Length = blockSizes[blockCount - 1];
			cOp.Type = 'M';
			bam.CigarData.push_back(cOp);
		}
	}
}


void MakeBamHeader(const string &genomeFile, RefVector &refs, string &header, 
                   map<string, int, std::less<string> > &chromToId) {
	
	// make a genome map of the genome file.
	GenomeFile genome(genomeFile);
	
	header += "@HD\tVN:1.0\tSO:unsorted\n";
	header += "@PG\tID:BEDTools_bedToBam\tVN:V";
	header += VERSION;
	header += "\n";
	
	int chromId = 0;
	vector<string> chromList = genome.getChromList();
	sort(chromList.begin(), chromList.end());
	
	// create a BAM header (@SQ) entry for each chrom in the BEDTools genome file.
	vector<string>::const_iterator genomeItr  = chromList.begin();
	vector<string>::const_iterator genomeEnd  = chromList.end();
	for (; genomeItr != genomeEnd; ++genomeItr) {		
		chromToId[*genomeItr] = chromId;
		chromId++;
		
		// add to the header text
		int size = genome.getChromSize(*genomeItr);
		string chromLine = "@SQ\tSN:" + *genomeItr + "\tAS:" + genomeFile + "\tLN:" + ToString(size) + "\n";
		header += chromLine;
		
		// create a chrom entry and add it to
		// the RefVector
		RefData chrom;
		chrom.RefName            = *genomeItr;
		chrom.RefLength          = size;
		chrom.RefHasAlignments   = false;
		refs.push_back(chrom);		
	}
}


/* Taken directly from the SAMTools spec
calculate bin given an alignment in [beg,end) (zero-based, half-close, half-open) */
int reg2bin(int beg, int end) {
	--end;
	if (beg>>14 == end>>14) return ((1<<15)-1)/7 + (beg>>14);
	if (beg>>17 == end>>17) return ((1<<12)-1)/7 + (beg>>17);
	if (beg>>20 == end>>20) return ((1<<9)-1)/7 + (beg>>20);
	if (beg>>23 == end>>23) return ((1<<6)-1)/7 + (beg>>23);
	if (beg>>26 == end>>26) return ((1<<3)-1)/7 + (beg>>26);
	return 0;
}	


