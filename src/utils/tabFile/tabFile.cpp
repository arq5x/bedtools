/*****************************************************************************
  tabFile.cpp

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licensed under the GNU General Public License 2.0 license.
******************************************************************************/
#include "lineFileUtilities.h"
#include "tabFile.h"

/*******************************************
Class methods
*******************************************/

// Constructor
TabFile::TabFile(const string &tabFile)
: _tabFile(tabFile)
{}

// Destructor
TabFile::~TabFile(void) {
}


void TabFile::Open(void) {
    if (_tabFile == "stdin" || _tabFile == "-") {
        _tabStream = &cin;
    }
    else {
        size_t foundPos;
        foundPos = _tabFile.find_last_of(".gz");
        // is this a GZIPPED TAB file?
        if (foundPos == _tabFile.size() - 1) {
            igzstream tabs(_tabFile.c_str(), ios::in);
            if ( !tabs ) {
                cerr << "Error: The requested file (" << _tabFile << ") could not be opened. Exiting!" << endl;
                exit (1);
            }
            else {
                // if so, close it (this was just a test)
                tabs.close();
                // now set a pointer to the stream so that we
                // can read the file later on.
                _tabStream = new igzstream(_tabFile.c_str(), ios::in);
            }
        }
        // not GZIPPED.
        else {

            ifstream tabs(_tabFile.c_str(), ios::in);
            // can we open the file?
            if ( !tabs ) {
                cerr << "Error: The requested file (" << _tabFile << ") could not be opened. Exiting!" << endl;
                exit (1);
            }
            else {
                // if so, close it (this was just a test)
                tabs.close();
                // now set a pointer to the stream so that we
                // can read the file later on.
                _tabStream = new ifstream(_tabFile.c_str(), ios::in);
            }
        }
    }
}


// Close the TAB file
void TabFile::Close(void) {
    if (_tabFile != "stdin" && _tabFile != "-") delete _tabStream;
}


TabLineStatus TabFile::GetNextTabLine(TAB_FIELDS &tabFields, int &lineNum) {

    // make sure there are still lines to process.
    // if so, tokenize, return the TAB_FIELDS.
    if (_tabStream->good() == true) {
        string tabLine;
        tabFields.reserve(20);

        // parse the tabStream pointer
        getline(*_tabStream, tabLine);
        lineNum++;

        // split into a string vector.
        Tokenize(tabLine, tabFields);

        // parse the line and validate it
        return parseTabLine(tabFields, lineNum);
    }

    // default if file is closed or EOF
    return TAB_INVALID;
}
