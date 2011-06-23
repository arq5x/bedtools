/*****************************************************************************
  tagBam.h

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0 license.
******************************************************************************/
#ifndef TAGBAM_H
#define TAGBAM_H

#include "bedFile.h"

#include "version.h"
#include "api/BamReader.h"
#include "api/BamWriter.h"
#include "api/BamAux.h"
#include "BamAncillary.h"
using namespace BamTools;

#include "bedFile.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>

using namespace std;

//************************************************
// Class methods and elements
//************************************************
class TagBam {

public:

    // constructor
    TagBam(const string &bamFile, const vector<string> &annoFileNames,
                const vector<string> &annoLabels, bool forceStrand);

    // destructor
    ~TagBam(void);

    // annotate the BAM file with all of the annotation files.
    void Tag();

private:

    // input files.
    string _bamFile;
    vector<string> _annoFileNames;
    vector<string> _annoLabels;

    // instance of a bed file class.
    BedFile *_bed;
    vector<BedFile*> _annoFiles;

    // do we care about strandedness when tagging?
    bool _forceStrand;

    // private function for reporting coverage information
    void ReportAnnotations();

    void OpenAnnoFiles();

    void CloseAnnoFiles();
    
    bool FindOneOrMoreOverlap(const BED &a, BedFile *bedFile);
};
#endif /* TAGBAM_H */
