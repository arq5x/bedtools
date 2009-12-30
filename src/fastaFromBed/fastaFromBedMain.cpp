/*****************************************************************************
  fastaFromBedMain.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "fastaFromBed.h"
#include "version.h"

using namespace std;

// define our program name
#define PROGRAM_NAME "fastaFromBed"


// define our parameter checking macro
#define PARAMETER_CHECK(param, paramLen, actualLen) (strncmp(argv[i], param, min(actualLen, paramLen))== 0) && (actualLen == paramLen)

// function declarations
void ShowHelp(void);

int main(int argc, char* argv[]) {

	// our configuration variables
	bool showHelp = false;

	// input files
	string fastaDbFile;
	string bedFile;

	// output files
	string fastaOutFile;

	// checks for existence of parameters
	bool haveFastaDb = false;
	bool haveBed = false;
	bool haveFastaOut = false;
	bool useNameOnly = false;

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

		if(PARAMETER_CHECK("-fi", 3, parameterLength)) {
			haveFastaDb = true;
			fastaDbFile = argv[i + 1];
			i++;
		} 
		else if(PARAMETER_CHECK("-fo", 3, parameterLength)) {
			haveFastaOut = true;
			fastaOutFile = argv[i + 1];
			i++;
		} 
		else if(PARAMETER_CHECK("-bed", 4, parameterLength)) {
			haveBed = true;
			bedFile = argv[i + 1];
			i++;
		}
		else if(PARAMETER_CHECK("-name", 5, parameterLength)) {
			useNameOnly = true;
			i++;
		}	
		else {
			cerr << "*****ERROR: Unrecognized parameter: " << argv[i] << " *****" << endl << endl;
			showHelp = true;
		}		
	}

	if (!haveFastaDb || !haveFastaOut || !haveBed) {
		showHelp = true;
	}
	
	if (!showHelp) {

		Bed2Fa *b2f = new Bed2Fa(useNameOnly, fastaDbFile, bedFile, fastaOutFile);
		b2f->ExtractDNA(); 
		return 0;
	}
	else {
		ShowHelp();
	}
}

void ShowHelp(void) {
	
	cerr << endl << "PROGRAM: " << PROGRAM_NAME << " (v" << VERSION << ")" << endl << endl;
	
	cerr << "AUTHOR:  Aaron Quinlan (aaronquinlan@gmail.com)" << endl << endl ;
	
	cerr << "SUMMARY: Extract DNA sequences into a fasta file based on BED coordinates." << endl << endl;

	cerr << "USAGE:   " << PROGRAM_NAME << " [OPTIONS] -fi -bed -fo " << endl << endl;

	cerr << "OPTIONS: " << endl;
	cerr << "\t\t-fi\tInput FASTA file" << endl;
	cerr << "\t\t-bed\tBED file of ranges to extract from -fi" << endl;
	cerr << "\t\t-fo\tOutput FASTA file" << endl;
	cerr << "\t\t-name\tUse the BED name field (#4) for the FASTA header" << endl;



	// end the program here
	exit(1);
	
}
