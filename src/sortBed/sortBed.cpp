// 
//  sortBed.cpp
//  BEDTools
//  
//  Created by Aaron Quinlan Spring 2009.
//  Copyright 2009 Aaron Quinlan. All rights reserved.
//
//  Summary:  Sorts a BED file in ascending order by chrom then by start position.
//
#include "lineFileUtilities.h"
#include "sortBed.h"

//
// Constructor
//
BedSort::BedSort(string &bedFile) {
	this->bedFile = bedFile;
	this->bed = new BedFile(bedFile);
}

//
// Destructor
//
BedSort::~BedSort(void) {
}


void BedSort::SortBed() {

	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	bed->loadBedFileIntoMapNoBin();

	// loop through each chromosome and merge their BED entries
	for (masterBedMapNoBin::iterator m = bed->bedMapNoBin.begin(); m != bed->bedMapNoBin.end(); ++m) {

		// bedList is already sorted by start position.
		vector<BED> bedList = m->second; 

		for (unsigned int i = 0; i < bedList.size(); ++i) {
			bed->reportBed(bedList[i]); cout << "\n";
		}
	}
}


void BedSort::SortBedBySizeAsc() {

	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	bed->loadBedFileIntoMapNoBin();

	vector<BED> masterList;
	masterList.reserve(1000000);
	
	// loop through each chromosome and merge their BED entries
	for (masterBedMapNoBin::iterator m = bed->bedMapNoBin.begin(); m != bed->bedMapNoBin.end(); ++m) {

		// bedList is already sorted by start position.
		vector<BED> bedList = m->second; 

		// add the entries from this chromosome to the current list
		for (unsigned int i = 0; i < m->second.size(); ++i) {
			masterList.push_back(m->second[i]);
		}
	}
	
	// sort the master list by size (asc.)
	sort(masterList.begin(), masterList.end(), sortBySizeAsc);
	
	// report the entries in ascending order
	for (unsigned int i = 0; i < masterList.size(); ++i) {
		bed->reportBed(masterList[i]); cout << "\n";
	}
}


void BedSort::SortBedBySizeDesc() {

	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	bed->loadBedFileIntoMapNoBin();

	vector<BED> masterList;
	masterList.reserve(1000000);
	
	// loop through each chromosome and merge their BED entries
	for (masterBedMapNoBin::iterator m = bed->bedMapNoBin.begin(); m != bed->bedMapNoBin.end(); ++m) {

		// bedList is already sorted by start position.
		vector<BED> bedList = m->second; 

		// add the entries from this chromosome to the current list
		for (unsigned int i = 0; i < m->second.size(); ++i) {
			masterList.push_back(m->second[i]);
		}
	}
	
	// sort the master list by size (asc.)
	sort(masterList.begin(), masterList.end(), sortBySizeDesc);
	
	// report the entries in ascending order
	for (unsigned int i = 0; i < masterList.size(); ++i) {
		bed->reportBed(masterList[i]); cout << "\n";
	}
}

void BedSort::SortBedByChromThenSizeAsc() {

	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	bed->loadBedFileIntoMapNoBin();

	// loop through each chromosome and merge their BED entries
	for (masterBedMapNoBin::iterator m = bed->bedMapNoBin.begin(); m != bed->bedMapNoBin.end(); ++m) {

		// bedList is already sorted by start position.
		vector<BED> bedList = m->second; 
		sort(bedList.begin(), bedList.end(), sortBySizeAsc);
		
		for (unsigned int i = 0; i < bedList.size(); ++i) {
			bed->reportBed(bedList[i]); cout << "\n";
		}
	}
}


void BedSort::SortBedByChromThenSizeDesc() {

	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	bed->loadBedFileIntoMapNoBin();

	// loop through each chromosome and merge their BED entries
	for (masterBedMapNoBin::iterator m = bed->bedMapNoBin.begin(); m != bed->bedMapNoBin.end(); ++m) {

		// bedList is already sorted by start position.
		vector<BED> bedList = m->second; 

		sort(bedList.begin(), bedList.end(), sortBySizeDesc);
		
		for (unsigned int i = 0; i < bedList.size(); ++i) {
			bed->reportBed(bedList[i]); cout << "\n";
		}
	}
}


void BedSort::SortBedByChromThenScoreAsc() {

	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	bed->loadBedFileIntoMapNoBin();

	if (bed->bedType >= 5) {
		// loop through each chromosome and merge their BED entries
		for (masterBedMapNoBin::iterator m = bed->bedMapNoBin.begin(); m != bed->bedMapNoBin.end(); ++m) {

			// bedList is already sorted by start position.
			vector<BED> bedList = m->second; 
			sort(bedList.begin(), bedList.end(), sortByScoreAsc);
			
			for (unsigned int i = 0; i < bedList.size(); ++i) {
				bed->reportBed(bedList[i]); cout << "\n";
			}
		}
	}
	else {
		cerr << "Error: Requested a sort by score, but your BED file does not appear to be in BED 5 format or greater.  Exiting." << endl;
		exit(1);
	}
}


void BedSort::SortBedByChromThenScoreDesc() {

	// load the "B" bed file into a map so
	// that we can easily compare "A" to it for overlaps
	bed->loadBedFileIntoMapNoBin();

	if (bed->bedType >= 5) {
		// loop through each chromosome and merge their BED entries
		for (masterBedMapNoBin::iterator m = bed->bedMapNoBin.begin(); m != bed->bedMapNoBin.end(); ++m) {

			// bedList is already sorted by start position.
			vector<BED> bedList = m->second; 
			sort(bedList.begin(), bedList.end(), sortByScoreDesc);
		
			for (unsigned int i = 0; i < bedList.size(); ++i) {
				bed->reportBed(bedList[i]); cout << "\n";
			}
		}
	}
	else {
		cerr << "Error: Requested a sort by score, but your BED file does not appear to be in BED 5 format or greater.  Exiting." << endl;
		exit(1);
	}
}

