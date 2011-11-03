/*****************************************************************************
  unionBedGraphs.cpp

  (c) 2010 - Assaf Gordon, CSHL
           - Aaron Quinlan, UVA
  Hall Laboratory
  Department of Biochemistry and Molecular Genetics
  University of Virginia
  aaronquinlan@gmail.com

  Licenced under the GNU General Public License 2.0 license.
******************************************************************************/
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "bedFile.h"
#include "multiIntersectBed.h"

using namespace std;


MultiIntersectBed::MultiIntersectBed(std::ostream& _output,
                            const vector<string>& _filenames,
                            const vector<string>& _titles,
                            bool _print_empty_regions,
                            const std::string& _genome_size_filename,
                            const std::string& _no_coverage_value   ) :
    filenames(_filenames),
    titles(_titles),
    output(_output),
    current_non_zero_inputs(0),
    print_empty_regions(_print_empty_regions),
    haveTitles(false),
    genome_sizes(NULL),
    no_coverage_value(_no_coverage_value)
{
    if (print_empty_regions) {
        assert(!_genome_size_filename.empty());

        genome_sizes = new GenomeFile(_genome_size_filename);
    }
    
    if (titles.size() > 0) {
        haveTitles = true;
    }
}


MultiIntersectBed::~MultiIntersectBed() {
    CloseFiles();
    if (genome_sizes) {
        delete genome_sizes;
        genome_sizes = NULL ;
    }
}


void MultiIntersectBed::MultiIntersect() {
    OpenFiles();

    // Add the first interval from each file
    for(size_t i = 0;i < input_files.size(); ++i)
        LoadNextItem(i);

    // Chromosome loop - once per chromosome
    do {
        // Find the first chromosome to use
        current_chrom = DetermineNextChrom();

        // Populate the queue with initial values from all files
        // (if they belong to the correct chromosome)
        for(size_t i = 0; i < input_files.size(); ++i)
            AddInterval(i);

        CHRPOS current_start = ConsumeNextCoordinate();

        // User wanted empty regions, and the first coordinate is not 0 - print a dummy empty coverage
        if (print_empty_regions && current_start > 0)
            PrintEmptyCoverage(0,current_start);

        // Intervals loop - until all intervals (of current chromosome) from all files are used.
        do {
            CHRPOS current_end = queue.top().coord;
            PrintCoverage(current_start, current_end);
            current_start = ConsumeNextCoordinate();
        } while (!queue.empty());

        // User wanted empty regions, and the last coordinate is not the last coordinate of the chromosome
            // print a dummy empty coverage
        if (print_empty_regions) {
            CHRPOS chrom_size = genome_sizes->getChromSize(current_chrom);
            if (current_start < chrom_size)
                PrintEmptyCoverage(current_start, chrom_size);
        }

    } while (!AllFilesDone());
}


CHRPOS MultiIntersectBed::ConsumeNextCoordinate() {
    assert(!queue.empty());

    CHRPOS new_position = queue.top().coord;
    do {
        IntervalItem item = queue.top();
        UpdateInformation(item);
        queue.pop();
    } while (!queue.empty() && queue.top().coord == new_position);

    return new_position;
}


void MultiIntersectBed::UpdateInformation(const IntervalItem &item) {
    // Update the depth coverage for this file

    // Which coordinate is it - start or end?
    switch (item.coord_type)
    {
    case START:
        current_depth[item.source_index] = 1;
        current_non_zero_inputs++;
        break;
    case END:
        //Read the next interval from this file
        AddInterval(item.source_index);
        current_depth[item.source_index] = 0;
        current_non_zero_inputs--;
        break;
    default:
        assert(0);
    }
}


void MultiIntersectBed::AddInterval(int index) {
    assert(static_cast<unsigned int>(index) < input_files.size());

    //This file has no more intervals
    if (current_item[index].chrom.empty())
        return;

    //If the next interval belongs to a different chrom, don't add it
    if (current_item[index].chrom!=current_chrom)
        return;

    const BED &bed(current_item[index]);

    IntervalItem start_item(index, START, bed.start);
    IntervalItem end_item(index, END, bed.end);

    queue.push(start_item);
    queue.push(end_item);

    LoadNextItem(index);
}


void MultiIntersectBed::PrintHeader() {
    output << "chrom\tstart\tend\tnum\tlist";
    if (titles.size() > 0) {
        for (size_t i = 0; i < titles.size(); ++i)
            output << "\t" << titles[i];
    }
    else {
        for (size_t i = 0;i < filenames.size(); ++i)
            output << "\t" << filenames[i];
    }
    output << endl;
}


void MultiIntersectBed::PrintCoverage(CHRPOS start, CHRPOS end) {
    if ( current_non_zero_inputs == 0 && ! print_empty_regions )
        return ;

    output << current_chrom << "\t"
        << start << "\t"
        << end   << "\t"
        << current_non_zero_inputs << "\t";
    
    ostringstream file_list_string;
    ostringstream file_bool_string;
    int depth_count = 0;
    for (size_t i = 0; i < current_depth.size(); ++i)
    {
        if (current_depth[i] > 0) {
            if (depth_count < current_non_zero_inputs - 1) {
                if (!haveTitles)
                    file_list_string << i+1 << ",";
                else 
                    file_list_string << titles[i] << ",";
            }
            else {
                if (!haveTitles)
                    file_list_string << i+1;
                else 
                    file_list_string << titles[i];
            }
            depth_count++;
        }
        file_bool_string << "\t" << current_depth[i];
    }
    if (current_non_zero_inputs > 0) {
        cout << file_list_string.str() << file_bool_string.str() << endl;
    }
    else {
        cout << "none" << file_bool_string.str() << endl;
    }
}


void MultiIntersectBed::PrintEmptyCoverage(CHRPOS start, CHRPOS end) {
    output << current_chrom << "\t"
        << start << "\t"
        << end   << "\t"
        << "0"   << "\t" << "none";
        
    for (size_t i=0;i<current_depth.size();++i)
        output << "\t0";

    output << endl;
}


void MultiIntersectBed::LoadNextItem(int index) {
    assert(static_cast<unsigned int>(index) < input_files.size());

    current_item[index].chrom="";

    BedFile *file = input_files[index];
    BED merged_bed;
    int lineNum = 0;
    //
    // TO DO: Do the mergeing on the fly.  How best to do this?
    // 
    // IDEA: Implement a Merge class with GetNextMerge element.
    //

    while (file->GetNextMergedBed(merged_bed, lineNum))
    {
        current_item[index] = merged_bed;
        break;
    }
}


bool MultiIntersectBed::AllFilesDone() {
    for (size_t i=0;i<current_item.size();++i)
        if (!current_item[i].chrom.empty())
            return false;
    return true;
}


string MultiIntersectBed::DetermineNextChrom() {
    string next_chrom;
    for (size_t i=0;i<current_item.size();++i) {
        if (current_item[i].chrom.empty())
            continue;

        if (next_chrom.empty())
            next_chrom = current_item[i].chrom;
        else
            if (current_item[i].chrom < next_chrom)
                next_chrom = current_item[i].chrom ;
    }
    return next_chrom;
}


void MultiIntersectBed::OpenFiles() {
    for (size_t i = 0; i < filenames.size(); ++i) {
        BedFile *file = new BedFile(filenames[i]);
        file->Open();
        input_files.push_back(file);
        current_depth.push_back(0);
    }
    current_item.resize(filenames.size());
}


void MultiIntersectBed::CloseFiles() {
    for (size_t i=0; i < input_files.size(); ++i) {
        BedFile *file = input_files[i];
        delete file;
        input_files[i] = NULL ;
    }
    input_files.clear();
}
