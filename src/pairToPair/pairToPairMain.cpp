/*****************************************************************************
  pairToPairMain.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "pairToPair.h"
#include "version.h"

using namespace std;

// define our program name
#define PROGRAM_NAME "pairToPair"

// define our parameter checking macro
#define PARAMETER_CHECK(param, paramLen, actualLen) (strncmp(argv[i], param, min(actualLen, paramLen))== 0) && (actualLen == paramLen)

// function declarations
void ShowHelp(void);

int main(int argc, char* argv[]) {

	// our configuration variables
	bool showHelp = false;

	// input files
	string bedAFile;
	string bedBFile;
	
	// input arguments
	float overlapFraction = 1E-9;
	string searchType = "both";

	// flags to track parameters
	bool haveBedA = false;
	bool haveBedB = false;
	bool haveSearchType = false;
	bool haveFraction = false;
	bool ignoreStrand = false;	

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

		if(PARAMETER_CHECK("-a", 2, parameterLength)) {
			if ((i+1) < argc) {
				haveBedA = true;
				bedAFile = argv[i + 1];
			}
			i++;
		}
		else if(PARAMETER_CHECK("-b", 2, parameterLength)) {
			if ((i+1) < argc) {
				haveBedB = true;
				bedBFile = argv[i + 1];
			}
			i++;
		}	
		else if(PARAMETER_CHECK("-type", 5, parameterLength)) {
			if ((i+1) < argc) {
				haveSearchType = true;
				searchType = argv[i + 1];
			}
			i++;
		}
		else if(PARAMETER_CHECK("-f", 2, parameterLength)) {
			haveFraction = true;
			overlapFraction = atof(argv[i + 1]);
			i++;
		}
		else if(PARAMETER_CHECK("-is", 3, parameterLength)) {
			ignoreStrand = true;
			i++;
		}
		else {
			cerr << endl << "*****ERROR: Unrecognized parameter: " << argv[i] << " *****" << endl << endl;
			showHelp = true;
		}		
	}


	// make sure we have both input files
	if (!haveBedA || !haveBedB) {
		cerr << endl << "*****" << endl << "*****ERROR: Need -a and -b files. " << endl << "*****" << endl;
		showHelp = true;
	}
	
	if (haveSearchType && (searchType != "neither") && (searchType != "both")) {
		cerr << endl << "*****" << endl << "*****ERROR: Request \"both\" or \"neither\"" << endl << "*****" << endl;
		showHelp = true;		
	}

	if (!showHelp) {

		PairToPair *bi = new PairToPair(bedAFile, bedBFile, overlapFraction, searchType, ignoreStrand);
		bi->DetermineBedPEInput();
		return 0;
	}
	else {
		ShowHelp();
	}
}

void ShowHelp(void) {

	cerr << endl << "Program: " << PROGRAM_NAME << " (v" << VERSION << ")" << endl;
	
	cerr << "Author:  Aaron Quinlan (aaronquinlan@gmail.com)" << endl;
	
	cerr << "Summary: Report overlaps between two paired-end BED files (BEDPE)." << endl << endl;

	cerr << "Usage:   " << PROGRAM_NAME << " [OPTIONS] -a <BEDPE> -b <BEDPE>" << endl << endl;

	cerr << "Options: " << endl;
	cerr << "\t-f\t"	    			<< "Minimum overlap required as fraction of A (e.g. 0.05)." << endl;
	cerr 								<< "\t\tDefault is 1E-9 (effectively 1bp)." << endl << endl;

	cerr << "\t-type \t"				<< "Approach to reporting overlaps between A and B." << endl;
	cerr 								<< "\t\tneither\t\tReport overlaps if neither end of A overlaps B." << endl << endl;

	cerr 								<< "\t\tboth\t\tReport overlaps if both ends of A overlap B." << endl;
	cerr									<< "\t\t\t\t- Default." << endl;

	cerr << "\t-is\t"	    			<< "Ignore strands when searching for overlaps." << endl;
	cerr 								<< "\t\t- By default, strands are enforced." << endl << endl;

	cerr << "Refer to the BEDTools manual for BEDPE format." << endl << endl;
		
	// end the program here
	exit(1);

}
