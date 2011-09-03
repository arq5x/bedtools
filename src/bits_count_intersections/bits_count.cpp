/*****************************************************************************
intersectBed.cpp

(c) 2009 - Aaron Quinlan
Hall Laboratory
Department of Biochemistry and Molecular Genetics
University of Virginia
aaronquinlan@gmail.com

Licenced under the GNU General Public License 2.0 license.
******************************************************************************/
#include "lineFileUtilities.h"
#include "bits_count.h"


/*
Constructor
*/
BitsCount::BitsCount(string bedAFile, string bedBFile, string genomeFile) {

    _bedAFile = bedAFile;
    _bedBFile = bedBFile;
    _genomeFile = genomeFile;
    
    // create new BED file objects for A and B
    _bedA = new BedFile(bedAFile);
    _bedB = new BedFile(bedBFile);
    _genome = new BedFile(genomeFile);
    
    CountOverlaps();
}


/*
Destructor
*/
BitsCount::~BitsCount(void) {
}





void BitsCount::CountOverlaps() {
    
    
    // !!! NOTE TO RYAN: Make this use the genomeFile class, which is just two columns, chr:length.
    
    // 1. Read the genome/universe file and compute offsets
    _genome->Open();
    BedLineStatus bedStatus;
    BED bed, nullBed;
    int lineNum = 0;
    CHRPOS curr_offset = 0;
    while ((bedStatus = _genome->GetNextBed(bed, lineNum)) != BED_INVALID) {
        if (bedStatus == BED_VALID) {
            _offsets[bed.chrom] = curr_offset;
            curr_offset += bed.end;
            bed = nullBed;
        }
    }
    // close up
    _genome->Close();
    

    // pload A and B into vectors
    _bedA->loadBedFileIntoVector();
    _bedB->loadBedFileIntoVector();

    vector<struct interval> A, B;
    
    // project A into U
    for (size_t i = 0; i < _bedA->bedList.size(); ++i) {
            struct interval ivl;
            CHRPOS projected_start = _offsets[_bedB->bedList[i].chrom] + _bedA->bedList[i].start;
            CHRPOS projected_end   = _offsets[_bedB->bedList[i].chrom] + _bedA->bedList[i].end;
            ivl.start = projected_start + 1;
            ivl.end   = projected_end;
            A.push_back(ivl);
    }
    
    // project B into U
    for (size_t i = 0; i < _bedB->bedList.size(); ++i) {
        struct interval ivl;
        CHRPOS projected_start = _offsets[_bedB->bedList[i].chrom] + _bedB->bedList[i].start;
        CHRPOS projected_end   = _offsets[_bedB->bedList[i].chrom] + _bedB->bedList[i].end;
        ivl.start = projected_start + 1;
        ivl.end   = projected_end;
        B.push_back(ivl);
    }
    
    // struct interval *A_ivls = (struct interval *) malloc(sizeof(struct interval) * A.size());
    // struct interval *B_ivls = (struct interval *) malloc(sizeof(struct interval) * B.size());
    // 
    // // UNSAFE. Better solution? Assumes contiguous memory.
    // memcpy(A_ivls, &A[0], sizeof(struct interval) * A.size() );
    // memcpy(B_ivls, &B[0], sizeof(struct interval) * B.size() );
    										     
    uint32_t tot_overlaps = count_intersections_bsearch_seq(&A[0], A.size(), &B[0], B.size());
    printf("%u\n", tot_overlaps);
}