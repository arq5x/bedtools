/*****************************************************************************
windowMaker.cpp

(c) 2009 - Aaron Quinlan
Hall Laboratory
Department of Biochemistry and Molecular Genetics
University of Virginia
aaronquinlan@gmail.com

Licenced under the GNU General Public License 2.0 license.
******************************************************************************/
#include "windowMaker.h"

WindowMaker::WindowMaker(string &fileName, ID_METHOD id_method, INPUT_FILE_TYPE input_file_type, uint32_t size, uint32_t step)
: _size(size)
, _step(step)
, _count(0)
, _window_method(FIXED_WINDOW_SIZE)
, _id_method(id_method)
{
    if (input_file_type==GENOME_FILE)
        MakeWindowsFromGenome(fileName);
    else
        MakeWindowsFromBED(fileName);
}

WindowMaker::WindowMaker(string &fileName, ID_METHOD id_method, INPUT_FILE_TYPE input_file_type, uint32_t count)
    : _size(0)
    , _step(0)
    , _count(count)
    , _window_method(FIXED_WINDOW_COUNT)
    , _id_method(id_method)
{
    if (input_file_type==GENOME_FILE)
        MakeWindowsFromGenome(fileName);
    else
        MakeWindowsFromBED(fileName);
}

WindowMaker::~WindowMaker(void) {}


void WindowMaker::MakeWindowsFromGenome(const string& genomeFileName) {

    GenomeFile *_genome = new GenomeFile(genomeFileName);

    // get a list of the chroms in the user's genome
    vector<string> chromList = _genome->getChromList();

    // process each chrom in the genome
    for (size_t c = 0; c < chromList.size(); ++c) {
        string chrom = chromList[c];

        BED bed(chrom,0,_genome->getChromSize(chrom));
        MakeBEDWindow(bed);
    }
}

void WindowMaker::MakeWindowsFromBED(string& bedFileName) {
    BedFile bf(bedFileName);
    bf.Open();

    BED bed;
    while (bf.GetNextBed(bed)) {
        if (bf._status == BED_VALID)
            MakeBEDWindow(bed);
    }
    bf.Close();
}

void WindowMaker::MakeBEDWindow(const BED& interval)
{
    if (_window_method==FIXED_WINDOW_SIZE)
        MakeFixedSizeWindow(interval);
    else
        MakeFixedCountWindow(interval);
}

void WindowMaker::MakeFixedSizeWindow(const BED& interval) {
    uint32_t i=1;
    for (uint32_t start = interval.start; start <= interval.end; start += _step, ++i) {
        string name = GenerateID(interval,i);
        if ((start + _size) <= interval.end) {
            cout << interval.chrom << "\t" << start << "\t" << start + _size << name << endl;
        }
        else if (start < interval.end) {
            cout << interval.chrom << "\t" << start << "\t" << interval.end << name << endl;
        }
    }
}

void WindowMaker::MakeFixedCountWindow(const BED& interval) {
    uint32_t interval_size = interval.end - interval.start ;
    uint32_t window_size = (interval_size-1)/_count + 1; // integer version of ceil(interval_size/_count)
    if (window_size==0 || interval_size==0)
        return;

    uint32_t i=1;
    for (uint32_t start = interval.start; start < interval.end; start += window_size, ++i) {
        string name = GenerateID(interval,i);
        uint32_t end = min(start + window_size,interval.end);
        cout << interval.chrom << "\t" << start << "\t" << end << name << endl;
    }
}

string WindowMaker::GenerateID(const BED& interval, uint32_t window_index) const {
    stringstream s;
    switch(_id_method) {
    case ID_SOURCE_ID:
         s << "\t" << interval.name;
         break;
    case ID_WINDOW_NUMBER:
         s << "\t" << window_index;
         break;
    case ID_SOURCE_ID_WINDOW_NUMBER:
         s << "\t" << interval.name << "_" << window_index;
    default:
    case ID_NONE:
         break;
    }
    return s.str();
}
