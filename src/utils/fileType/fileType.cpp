/*****************************************************************************
fileType.cpp

(c) 2009 - Aaron Quinlan
Hall Laboratory
Department of Biochemistry and Molecular Genetics
University of Virginia
aaronquinlan@gmail.com

Licensed under the GNU General Public License 2.0 license.
******************************************************************************/

#include "fileType.h"


/*
returns TRUE if the file is a regular file:
not a pipe/device.

This implies that the file can be opened/closed/seek'd multiple times without losing information
*/
bool isRegularFile(const string& filename) {
    struct stat buf ;
    int i;

    i = stat(filename.c_str(), &buf);
    if (i!=0) {
        cerr << "Error: can't determine file type of '" << filename << "': " << strerror(errno) << endl;
        exit(1);
    }
    if (S_ISREG(buf.st_mode))
        return true;

    return false;
}


/*
returns TRUE if the file has a GZIP header.
Should only be run on regular files.
*/
bool isGzipFile(istream *file) {
    //see http://www.gzip.org/zlib/rfc-gzip.html#file-format
    struct  {
        unsigned char id1;
//        unsigned char id2;
//        unsigned char cm;
    } gzip_header;

    if (!file->read((char*)&gzip_header, sizeof(gzip_header))) {
        return false;
    }

    if ( gzip_header.id1 == 0x1f )
//        &&
//        gzip_header.id2 == 0x8b
//        &&
//        gzip_header.cm == 8 )
    {
        return true;
    }
    file->putback(gzip_header.id1);
    return false;
}
