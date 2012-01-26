/*****************************************************************************
windowMaker.h

(c) 2009 - Aaron Quinlan
Hall Laboratory
Department of Biochemistry and Molecular Genetics
University of Virginia
aaronquinlan@gmail.com

Licenced under the GNU General Public License 2.0 license.
******************************************************************************/
#include "genomeFile.h"
#include "bedFile.h"

using namespace std;


//************************************************
// Class methods and elements
//************************************************
class WindowMaker {

public:
    enum INPUT_FILE_TYPE {
        GENOME_FILE,
        BED_FILE
    };
    enum WINDOW_METHOD {
        FIXED_WINDOW_SIZE,
        FIXED_WINDOW_COUNT
    };

    // constructor
    WindowMaker(string &fileName, INPUT_FILE_TYPE input_file_type, uint32_t count);
    WindowMaker(string &fileName, INPUT_FILE_TYPE input_file_type, uint32_t size, uint32_t step);

    // destructor
    ~WindowMaker(void);

    void MakeWindowsFromGenome(const string& genomeFileName);
    void MakeWindowsFromBED(string& bedFileName);

private:
    uint32_t _size;
    uint32_t _step;
    uint32_t _count;
    WINDOW_METHOD _window_method;

    void MakeBEDWindow(const BED& interval);

    void MakeFixedSizeWindow(const BED& interval);
    void MakeFixedCountWindow(const BED& interval);
};