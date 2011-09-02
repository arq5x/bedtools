#ifndef __SEQ_FILE_H__
#define __SEQ_FILE_H__
#include <string>
#include <fstream>
using namespace std;

class fasta_file {

public:
	fasta_file(string file_name, size_t line_len);
	~fasta_file();
	string get_seq(int start, int end);
private:
	string trim(string str, int most_breaks);
	char *buffer;
	ifstream is;
	int data_start;
	size_t len;
};

#endif
