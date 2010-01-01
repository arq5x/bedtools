/*****************************************************************************
  complementBedMain.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "complementBed.h"
#include "version.h"

using namespace std;

// define our program name
#define PROGRAM_NAME "complementBed"


// define our parameter checking macro
#define PARAMETER_CHECK(param, paramLen, actualLen) (strncmp(argv[i], param, min(actualLen, paramLen))== 0) && (actualLen == paramLen)

// function declarations
void ShowHelp(void);

int main(int argc, char* argv[]) {

	// our configuration variables
	bool showHelp = false;

	// input files
	string bedFile;
	string genomeFile;

	bool haveBed = false;
	bool haveGenome = false;

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
			haveBed = true;
			bedFile = argv[i + 1];
			i++;
		}
		else if(PARAMETER_CHECK("-g", 2, parameterLength)) {
			haveGenome = true;
			genomeFile = argv[i + 1];
			i++;
		}
		else {
		  cerr << endl << "*****ERROR: Unrecognized parameter: " << argv[i] << " *****" << endl << endl;
			showHelp = true;
		}		
	}

	// make sure we have both input files
	if (!haveBed || !haveGenome) {
	  cerr << endl << "*****" << endl << "*****ERROR: Need -i BED file and -g Genome file. " << endl << "*****" << endl;
	  showHelp = true;
	}
	if (!showHelp) {
		BedComplement *bc = new BedComplement(bedFile, genomeFile);
		bc->ComplementBed();
		return 0;
	}
	else {
		ShowHelp();
	}
}

void ShowHelp(void) {
	
	cerr << endl << "Program: " << PROGRAM_NAME << " (v" << VERSION << ")" << endl;
	
	cerr << "Author:  Aaron Quinlan (aaronquinlan@gmail.com)" << endl;
	
	cerr << "Summary: Returns the base pair complement of a BED file." << endl << endl;

	cerr << "Usage:   " << PROGRAM_NAME << " [OPTIONS] -i <bed> -g <genome>" << endl << endl;
	
	cerr << "Notes: " << endl;
	cerr << "\t(1)  The genome file should tab delimited and structured as follows:" << endl;
	cerr << "\t     <chromName><TAB><chromSize>" << endl << endl;
	cerr << "\tFor example, Human (hg19):" << endl;
	cerr << "\tchr1\t249250621" << endl;
	cerr << "\tchr2\t243199373" << endl;
	cerr << "\t..." << endl;
	cerr << "\tchr18_gl000207_random\t4262" << endl << endl;
	
	cerr << "Tips: " << endl;
	cerr << "\tOne can use the UCSC Genome Browser's MySQL database to extract" << endl;
	cerr << "\tchromosome sizes. For example, H. sapiens:" << endl << endl;
	cerr << "\tmysql --user=genome --host=genome-mysql.cse.ucsc.edu -A -e /" << endl;
	cerr << "\t\"select chrom, size from hg19.chromInfo\"  > hg19.genome" << endl << endl;

	exit(1);
	
}
