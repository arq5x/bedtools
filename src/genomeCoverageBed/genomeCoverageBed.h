/*****************************************************************************
genomeCoverage.h

(c) 2009 - Aaron Quinlan
Hall Laboratory
Department of Biochemistry and Molecular Genetics
University of Virginia
aaronquinlan@gmail.com

Licenced under the GNU General Public License 2.0+ license.
******************************************************************************/
#include "bedFile.h"

#include "BamReader.h"
#include "BamAux.h"
using namespace BamTools;

#include <vector>
#include <iostream>
#include <fstream>
using namespace std;


//***********************************************
// Typedefs
//***********************************************
typedef map<int, DEPTH, less<int> >          depthMap;
typedef map<string, depthMap, less<string> > chromDepthMap;

typedef map<int, unsigned int, less<int> >   histMap;
typedef map<string, histMap, less<string> >  chromHistMap;

//************************************************
// Class methods and elements
//************************************************
class BedGenomeCoverage {

public:

	// constructor 
	BedGenomeCoverage(string &bedFile, string &genomeFile, bool &eachBase, bool &startSites, 
		bool &bedGraph, int &max, bool &bamInput);

	// destructor
	~BedGenomeCoverage(void);

	void CoverageBed(istream &bedInput);

	void CoverageBam(string bamFile);

	void ReportChromCoverage(const vector<DEPTH> &, int &chromSize, string &chrom, chromHistMap&);

	void ReportGenomeCoverage(map<string, int> &chromSizes, chromHistMap &chromDepthHist);

	void ReportChromCoverageBedGraph(const vector<DEPTH> &chromCov, int &chromSize, string &chrom);

	void DetermineBedInput();

private:

	string bedFile;
	string genomeFile;
	bool bamInput;
	bool eachBase;
	bool startSites;
	bool bedGraph;
	int max;

	// The BED file from which to compute coverage.
	BedFile *bed;
	chromDepthMap chromCov;
};
