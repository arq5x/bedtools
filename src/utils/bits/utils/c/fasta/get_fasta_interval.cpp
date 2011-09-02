#include <string>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

#include "fasta_file.h"


int main (int argc, char** argv) {

	if (argc < 4) {
		cerr << "usage:\t" <<
			argv[0] <<
			"<start> <end> <file> <line size>" << 
			endl;
		return 1;
	}

	int start = atoi(argv[1]), end = atoi(argv[2]), size = atoi(argv[4]);

	fasta_file f(argv[3], size);

	cout << f.get_seq(start, end) << endl;

	return 0;
}
