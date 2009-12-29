/*****************************************************************************
  windowMain.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "windowBed.h"
#include "version.h"

using namespace std;


// define the version
#define PROGRAM_NAME "windowBed"

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
	int leftSlop = 1000;
	int rightSlop = 1000;

	bool haveBedA = false;
	bool haveBedB = false;
	bool noHit = false;
	bool anyHit = false;
	bool writeCount = false;
	bool haveSlop = false;
	bool haveLeft = false;
	bool haveRight = false;
	bool strandWindows = false;
	bool matchOnStrand = false;

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
		else if(PARAMETER_CHECK("-u", 2, parameterLength)) {
			anyHit = true;
		}
		else if(PARAMETER_CHECK("-c", 2, parameterLength)) {
			writeCount = true;
		}
		else if (PARAMETER_CHECK("-v", 2, parameterLength)) {
			noHit = true;
		}
		else if (PARAMETER_CHECK("-sw", 3, parameterLength)) {
			strandWindows = true;
		}
		else if (PARAMETER_CHECK("-sm", 3, parameterLength)) {
			matchOnStrand = true;
		}
		else if (PARAMETER_CHECK("-w", 2, parameterLength)) {
			haveSlop = true;
			leftSlop = atoi(argv[i + 1]);
			rightSlop = leftSlop;
			i++;
		}
		else if (PARAMETER_CHECK("-l", 2, parameterLength)) {
			haveLeft = true;
			leftSlop = atoi(argv[i + 1]);
			i++;
		}
		else if (PARAMETER_CHECK("-r", 2, parameterLength)) {
			haveRight = true;
			rightSlop = atoi(argv[i + 1]);
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

	if (anyHit && noHit) {
		cerr << endl << "*****" << endl << "*****ERROR: Request either -u OR -v, not both." << endl << "*****" << endl;
		showHelp = true;
	}
	
	if (anyHit && writeCount) {
		cerr << endl << "*****" << endl << "*****ERROR: Request either -u OR -c, not both." << endl << "*****" << endl;
		showHelp = true;
	}

	if (haveLeft && (leftSlop < 0)) {
		cerr << endl << "*****" << endl << "*****ERROR: Upstream window (-l) must be positive." << endl << "*****" << endl;
		showHelp = true;
	}
	
	if (haveRight && (rightSlop < 0)) {
		cerr << endl << "*****" << endl << "*****ERROR: Downstream window (-r) must be positive." << endl << "*****" << endl;
		showHelp = true;
	}
	
	if (haveSlop && (haveLeft || haveRight)) {
		cerr << endl << "*****" << endl << "*****ERROR: Cannot choose -w with -l or -r.  Either specify -l and -r or specify solely -w" << endl << "*****" << endl;
		showHelp = true;		
	}
	
	if ((haveLeft && !haveRight) || (haveRight && !haveLeft)) {
		cerr << endl << "*****" << endl << "*****ERROR: Please specify both -l and -r." << endl << "*****" << endl;
		showHelp = true;		
	}
	
	if (!showHelp) {
		BedWindow *bi = new BedWindow(bedAFile, bedBFile, leftSlop, rightSlop, anyHit, noHit, writeCount, strandWindows, matchOnStrand);
		bi->DetermineBedInput();
		return 0;
	}
	else {
		ShowHelp();
	}
}

void ShowHelp(void) {

	cerr << "===============================================" << endl;
	cerr << " " <<PROGRAM_NAME << " v" << VERSION << endl ;
	cerr << " Aaron Quinlan, Ph.D. (aaronquinlan@gmail.com)  " << endl ;
	cerr << " Hall Laboratory, University of Virginia" << endl;
	cerr << "===============================================" << endl << endl;
	cerr << "SUMMARY: Examines a \"window\" around each feature in A and" << endl;
	cerr << "  reports all features in B that overlap the window. For each" << endl;
	cerr << "  overlap the entire entry in A and B are reported." << endl << endl;

	cerr << "USAGE: " << PROGRAM_NAME << " [OPTIONS] -a <a.bed> -b <b.bed>" << endl << endl;

	cerr << "OPTIONS: " << endl;
	cerr << "  " << "-w\t"		<< "Base pairs added upstream and downstream of each entry" << endl;
	cerr						<< "\tin A when searching for overlaps in B." << endl;
	cerr						<< "\t  - Creates symterical \"windows\" around A." << endl;		
	cerr						<< "\t  - Default is 1000 bp." << endl << endl;
	
	cerr << "  " << "-l\t"		<< "Base pairs added upstream (left of) of each entry" << endl;
	cerr						<< "\tin A when searching for overlaps in B." << endl;	
	cerr						<< "\t  - Allows one to define assymterical \"windows\"." << endl;
	cerr						<< "\t  - Default is 1000 bp." << endl << endl;

	cerr << "  " << "-r\t"		<< "Base pairs added downstream (right of) of each entry" << endl;
	cerr						<< "\tin A when searching for overlaps in B." << endl;	
	cerr						<< "\t  - Allows one to define assymterical \"windows\"." << endl;
	cerr						<< "\t  - Default is 1000 bp." << endl << endl;
		

	cerr << "  " << "-sw\t"     << "Define -l and -r based on strand.  For example if used, -l 500" << endl;
	cerr 						<< "\tfor a negative-stranded feature will add 500 bp downstream." << endl;
	cerr						<< "\t  - Default = disabled." << endl << endl;	

	cerr << "  " << "-sm\t"     << "Only report hits in B that overlap A on the same strand." << endl;
	cerr						<< "\t  - By default, overlaps are reported without respect to strand." << endl << endl;	

	cerr << "  " << "-u\t"      << "Write the original A entry _once_ if _any_ overlaps found in B." << endl;
	cerr 						<< "\t  - In other words, just report the fact >=1 hit was found." << endl << endl;

	cerr << "  " << "-c\t"		<< "For each entry in A, report the number of overlaps with B." << endl; 
	cerr 						<< "\t  - Reports 0 for A entries that have no overlap with B." << endl;
	cerr						<< "\t  - Overlaps restricted by -f." << endl << endl;

	cerr << "  " << "-v\t"      << "Only report those entries in A that have _no overlaps_ with B." << endl;
	cerr 						<< "\t  - Similar to \"grep -v.\"" << endl << endl;

	// end the program here
	exit(1);

}
