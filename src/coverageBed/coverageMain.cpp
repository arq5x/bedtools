/*****************************************************************************
  coverageMain.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "coverageBed.h"
#include "version.h"

using namespace std;

// define the version
#define PROGRAM_NAME "coverageBed"

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
	bool forceStrand = false;

	bool haveBedA = false;
	bool haveBedB = false;


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
			haveBedA = true;
			bedAFile = argv[i + 1];
			i++;
		}
		else if(PARAMETER_CHECK("-b", 2, parameterLength)) {
			haveBedB = true;
			bedBFile = argv[i + 1];
			i++;
		}
		else if (PARAMETER_CHECK("-s", 2, parameterLength)) {
			forceStrand = true;
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

	if (!showHelp) {

		BedCoverage *bg = new BedCoverage(bedAFile, bedBFile, forceStrand);
		bg->DetermineBedInput();
		return 0;
	}
	else {
		ShowHelp();
	}
}

void ShowHelp(void) {
	
	cerr << endl << "Program: " << PROGRAM_NAME << " (v" << VERSION << ")" << endl;
	
	cerr << "Author:  Aaron Quinlan (aaronquinlan@gmail.com)" << endl;
	
	cerr << "Summary: Returns the depth and breadth of coverage of features from A" << endl;
	cerr << "\t\t on the intervals in B." << endl << endl;
	
	cerr << "Usage:   " << PROGRAM_NAME << " [OPTIONS] -a <a.bed> -b <b.bed>" << endl << endl;

	cerr << "Output:  " << endl;
	cerr << "\t" << " After each entry in B, reports: " << endl; 
	cerr << "\t  1) The number of features in A that overlapped the B interval." << endl;
	cerr << "\t  2) The number of bases in B that had non-zero coverage." << endl;
	cerr << "\t  3) The length of the entry in B." << endl;
	cerr << "\t  4) The fraction of bases in B that had non-zero coverage." << endl;
	
	exit(1);

}
