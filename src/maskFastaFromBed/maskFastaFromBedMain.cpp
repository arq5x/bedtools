/*****************************************************************************
  maskFastaFromBedMain.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0 license.
******************************************************************************/
#include "maskFastaFromBed.h"

using namespace std;

// define our program name
#define PROGRAM_NAME "bedtools maskfasta"


// define our parameter checking macro
#define PARAMETER_CHECK(param, paramLen, actualLen) (strncmp(argv[i], param, min(actualLen, paramLen))== 0) && (actualLen == paramLen)

// function declarations
void maskfastafrombed_help(void);

int maskfastafrombed_main(int argc, char* argv[]) {

    // our configuration variables
    bool showHelp = false;

    // input files
    string fastaInFile;
    string bedFile;

    // output files
    string fastaOutFile;

    // defaults for parameters
    bool haveFastaIn  = false;
    bool haveBed      = false;
    bool haveFastaOut = false;
    bool softMask     = false;
    char maskChar     = 'N';

    // check to see if we should print out some help
    if(argc <= 1) showHelp = true;

    for(int i = 1; i < argc; i++) {
        int parameterLength = (int)strlen(argv[i]);

        if((PARAMETER_CHECK("-h", 2, parameterLength)) ||
        (PARAMETER_CHECK("--help", 5, parameterLength))) {
            showHelp = true;
        }
    }

    if(showHelp) maskfastafrombed_help();

    // do some parsing (all of these parameters require 2 strings)
    for(int i = 1; i < argc; i++) {

        int parameterLength = (int)strlen(argv[i]);

        if(PARAMETER_CHECK("-fi", 3, parameterLength)) {
            if ((i+1) < argc) {
                haveFastaIn = true;
                fastaInFile = argv[i + 1];
                i++;
            }
        }
        else if(PARAMETER_CHECK("-fo", 3, parameterLength)) {
            if ((i+1) < argc) {
                haveFastaOut = true;
                fastaOutFile = argv[i + 1];
                i++;
            }
        }
        else if(PARAMETER_CHECK("-bed", 4, parameterLength)) {
            if ((i+1) < argc) {
                haveBed = true;
                bedFile = argv[i + 1];
                i++;
            }
        }
        else if(PARAMETER_CHECK("-soft", 5, parameterLength)) {
            softMask = true;
        }
        else if(PARAMETER_CHECK("-mc", 3, parameterLength)) {
            if ((i+1) < argc) {
                string mask = argv[i + 1];
                if (mask.size() > 1) {
                    cerr << "*****ERROR: The mask character (-mc) should be a single character.*****" << endl << endl;
                    showHelp = true;
                }
                else {
                    maskChar = mask[0];
                }
                i++;
            }
        }
        else {
            cerr << "*****ERROR: Unrecognized parameter: " << argv[i] << " *****" << endl << endl;
            showHelp = true;
        }
    }

    if (!haveFastaIn || !haveFastaOut || !haveBed) {
        showHelp = true;
    }

    if (!showHelp) {

        MaskFastaFromBed *maskFasta = new MaskFastaFromBed(fastaInFile, bedFile, fastaOutFile, softMask, maskChar);
        delete maskFasta;
    }
    else {
        maskfastafrombed_help();
    }
    return 0;
}

void maskfastafrombed_help(void) {

    cerr << "\nTool:    bedtools maskfasta (aka maskFastaFromBed)" << endl;
    
    cerr << "Summary: Mask a fasta file based on feature coordinates." << endl << endl;

    cerr << "Usage:   " << PROGRAM_NAME << " [OPTIONS] -fi <fasta> -fo <fasta> -bed <bed/gff/vcf>" << endl << endl;

    cerr << "Options:" << endl;
    cerr << "\t-fi\tInput FASTA file" << endl;
    cerr << "\t-bed\tBED/GFF/VCF file of ranges to mask in -fi" << endl;
    cerr << "\t-fo\tOutput FASTA file" << endl;
    cerr << "\t-soft\tEnforce \"soft\" masking.  That is, instead of masking with Ns," << endl;
    cerr << "\t\tmask with lower-case bases." << endl;
    cerr << "\t-mc\tReplace masking character.  That is, instead of masking" << endl;
    cerr << "\t\twith Ns, use another character." << endl << endl;
    // end the program here
    exit(1);

}
