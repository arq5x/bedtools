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

    _bedAFile            = bedAFile;
    _bedBFile            = bedBFile;
    _genomeFile          = genomeFile;
    
    // create new BED file objects for A and B
    _bedA   = new BedFile(bedAFile);
    _bedB   = new BedFile(bedBFile);
    _genome = new BedFile(genomeFile);
    
    CountOverlaps();
}


/*
    Destructor
*/
BitsCount::~BitsCount(void) {
}





void BitsCount::CountOverlaps() {
    
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
    
    vector<CHRPOS> A_s_vec, A_e_vec, B_s_vec, B_e_vec;


    // project A into U
    _bedA->Open();
    lineNum = 0;
    uint32_t A_card = 0;  // cardinality of A
    while ((bedStatus = _bedA->GetNextBed(bed, lineNum)) != BED_INVALID) {
        if (bedStatus == BED_VALID) {
            CHRPOS projected_start = _offsets[bed.chrom] + bed.start;
            CHRPOS projected_end   = _offsets[bed.chrom] + bed.end;
            A_s_vec.push_back(projected_start + 1);
            A_e_vec.push_back(projected_end);
            bed = nullBed;
            A_card++;
        }
    }
    _bedA->Close();
    
    
    // project A into U    
    _bedB->Open();
    lineNum = 0;
    uint32_t B_card = 0;  // cardinality of B
    while ((bedStatus = _bedB->GetNextBed(bed, lineNum)) != BED_INVALID) {
        if (bedStatus == BED_VALID) {
            CHRPOS projected_start = _offsets[bed.chrom] + bed.start;
            CHRPOS projected_end   = _offsets[bed.chrom] + bed.end;
            B_s_vec.push_back(projected_start + 1);
            B_e_vec.push_back(projected_end);
            bed = nullBed;
            B_card++;
        }
    }
    _bedB->Close();
    
    sort(B_s_vec.begin(), B_s_vec.end());
    sort(B_e_vec.begin(), B_e_vec.end());
    
    uint32_t *A_starts = (uint32_t *) malloc(sizeof(uint32_t) * A_card);
    uint32_t *A_ends   = (uint32_t *) malloc(sizeof(uint32_t) * A_card);
    uint32_t *B_starts = (uint32_t *) malloc(sizeof(uint32_t) * B_card);
    uint32_t *B_ends   = (uint32_t *) malloc(sizeof(uint32_t) * B_card);
    
    // UNSAFE.  Better solution? Assumes contiguous memory.            
    memcpy(A_starts, &A_s_vec[0], sizeof(uint32_t) * A_card );
    memcpy(A_ends,   &A_e_vec[0], sizeof(uint32_t) * A_card );
    memcpy(B_starts, &B_s_vec[0], sizeof(uint32_t) * B_card );
    memcpy(B_ends,   &B_e_vec[0], sizeof(uint32_t) * B_card );
    
    uint32_t tot_overlaps = count_intersections_bsearch_seq(A_starts, A_ends, A_card, B_starts, B_ends, B_card); 
    printf("%u\n", tot_overlaps);
}

