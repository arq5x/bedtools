/*****************************************************************************
  chromsweepBed.h

  (c) 2009 - Aaron Quinlan
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0 license.
******************************************************************************/
#ifndef CHROMSWEEP_H
#define CHROMSWEEP_H

#include "bedFile.h"
#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <stdlib.h>
using namespace std;


// we want to fail on any unsupported data types
// this constructor will be called anytime a non-BED, non-BamAlignment DataWrapper<T> is accidentally coded
template<typename T>
class DataWrapper {
    public: DataWrapper(const T& value) { assert(false); } 
};

// otherwise we want to provide template specializations that will wrap the underlying data structures
// providing the basic primitives you'll need for the algorithm (refId, start, stop, whatever)

// ARQ Notes: 

// so here's a BamAlignment wrapper
template<>
class DataWrapper<BamAlignment> {

    public:
        typedef BamAlignment base_type;

    public:
        DataWrapper(const base_type& value = base_type()) 
        : m_data(value) 
        { }
        
        void setChrom(const string& name) { m_chrom = name; }
        inline string chrom(void) const { return m_chrom; }
        inline int start(void)    const { return m_data.Position; }
        inline int stop(void)     const { return m_data.GetEndPosition(); }

        inline base_type& data(void) { return m_data; }

        static string typeId(void) { return "DataWrapper<"+base_type::typeId()+">"; }
    private:
        string m_chrom;
        base_type m_data;
};

// and a BED wrapper
template<>
class DataWrapper<BED> {

    public:
        typedef BED base_type;

    public:
        DataWrapper(const base_type& value = base_type()) 
        : m_data(value) { }
        inline string chrom(void) { return m_data.chrom; }
        inline int start(void) { return m_data.start; }
        inline int stop(void)  { return m_data.stop; }

        inline base_type& data(void) { return m_data; }

        static string typeId(void) { return "DataWrapper<"+base_type::typeId()+">"; }
    private:
        base_type m_data;
};


// again here, we want to fail on any unsupported input types
template<typename T>
class InputWrapper {
    public: InputWrapper(T& input) { assert(false); } 
};

// the input wrapper will provide a common interface, but uses template 
// specialization to provide the API-specific "magic" under the hood

// BamReader wrapper
template<>
class InputWrapper<BamReader> {

    public:
        typedef BamReader                  input_type;
        typedef BamAlignment               data_basetype;
        typedef DataWrapper<data_basetype> data_type;

    public:
        InputWrapper(string input)
            : m_fileName(input) 
        { 
            m_bamReader = new BamReader;
            Open();
        }
        
        void Open() 
        {
            m_bamReader->Open(m_fileName);
        }
        void Close(void) {
            m_bamReader->Close();
        }
        bool GetNext(data_type& bam) {
            return m_bamReader->GetNextAlignment(bam.data());

            // get reference name using m_bamReader's (probably cached) references at index bam.data().RefID
            // the next line is very ugly & unsafe, but the logic is more what I'm trying to convey for now
            const string& name = m_bamReader.GetReferenceData()[bam.data().RefID].RefName;
            bam.setChrom(name);
            return true;
        }

        static string typeId(void) { return "InputWrapper<"+input_type::typeId()+">"; }
    private:
        input_type* m_bamReader;
        string m_fileName;
};


// BedFile wrapper
template<>
class InputWrapper<BedFile> {

    public:
        typedef BedFile                    input_type;
        typedef BED                        data_basetype;
        typedef DataWrapper<data_basetype> data_type;

    public:
        InputWrapper(string input)
            : m_fileName(input) 
        { 
            m_bedFile = new BedFile;
            Open();
        }
        void Open() 
        {
            m_bedFile->Open(m_fileName);
        }
        void Close(void) {
            m_bedFile->Close();
        }
        bool GetNext(data_type& bed) {
            return ( m_bedFile->GetNextBed(bed.data()) != BED_INVALID ); 
        }

        static string typeId(void) { return "InputWrapper<"+input_type::typeId()+">"; }        
    private:
        input_type* m_bedFile;
        string m_fileName;
};


// **** and now, here's where we tying the wrappers together ****

// again, force the code to use only types we specify
template<typename InputType, typename DataType>
class ApiPolicy {
    public: ApiPolicy(void) { assert(false); } // fail on any unsupported API types
};


// provides an BAM-specific API policy, which basically just exists to give you the typenames 
// of the wrappers that you need to run the algorithm
template<>
class ApiPolicy<BamReader, BamAlignment> {

    public:
        typedef BamReader                    input_basetype;
        typedef BamAlignment                 data_basetype; 
        typedef InputWrapper<input_basetype> input_type;
        typedef DataWrapper<data_basetype>   data_type;
};

// same here, just for BED data
template<>
class ApiPolicy<BedFile, BED> {

    public:
        typedef BedFile                      input_basetype;
        typedef BED                          data_basetype;
        typedef InputWrapper<input_basetype> input_type;
        typedef DataWrapper<data_basetype>   data_type;
};


/*
The chromsweep algorithm
*/
template<typename SourcePolicy, typename QueryPolicy>
class ChromSweep {
    
public:
        typedef typename SourcePolicy::input_type source_input_type;
        typedef typename SourcePolicy::data_type  source_data_type;
        typedef typename QueryPolicy::input_type  query_input_type;
        typedef typename QueryPolicy::data_type   query_data_type;

// public interface.
public:

    // A is the query and B is the database
    
    // constructor using existing BedFile pointers
    ChromSweep(BedFile *bedA, BedFile *bedB);
    
    // constructor using filenames
    ChromSweep(string &bedAFile, string &bedBFile);
    
    // destructor
    ~ChromSweep(void);
    
    // loads next (a pair) with the current query and it's overlaps
    //   next.first is the current query interval
    //   next.second is a vector of the current query's hits.
    // returns true if overlap
    bool Next(pair<BED, vector<BED> > &next);
    
    // Usage:
    //     ChromSweep sweep = ChromSweep(_bedA, _bedB);
    //     pair<BED, vector<BED> > hit_set;
    //     while (sweep.Next(hit_set)) 
    //     {
    //        // magic happens here!
    //        processHits(hit_set.first, hit_set.second);
    //     }
    
// private variables.
private:

    // instances of a bed file class.
    BedFile *_bedA, *_bedB;

    vector<BED> _cache;
    vector<BED> _hits;
    queue< pair<BED, vector<BED> > > _results;
    
    BED _nullBed;
    
    // variables for the current query and db entries.
    BED _curr_qy, _curr_db;
    BedLineStatus _qy_status, _db_status;
    int _qy_lineNum, _db_lineNum;

// private methods.
private:
    
    void ScanCache();
    void ChromCheck();
};

#endif /* CHROMSWEEP_H */
