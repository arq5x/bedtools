// ***************************************************************************
// BGZF.cpp (c) 2009 Derek Barnett, Michael Str�mberg
// Marth Lab, Department of Biology, Boston College
// All rights reserved.
// ---------------------------------------------------------------------------
// Last modified: 19 July 2010 (DB)
// ---------------------------------------------------------------------------
// BGZF routines were adapted from the bgzf.c code developed at the Broad
// Institute.
// ---------------------------------------------------------------------------
// Provides the basic functionality for reading & writing BGZF files
// ***************************************************************************

#include <algorithm>
#include "BGZF.h"
using namespace BamTools;
using std::string;
using std::min;


size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written;
    written = fwrite(ptr, size, nmemb, stream);
    return written;
}


 
enum fcurl_type_e { CFTYPE_NONE=0, CFTYPE_FILE=1, CFTYPE_CURL=2 };
 
struct fcurl_data
{
    enum fcurl_type_e type;     /* type of handle */ 
    union {
        CURL *curl;
        FILE *file;
    } handle;                   /* handle */ 
 
    char *buffer;               /* buffer to store cached data*/ 
    int buffer_len;             /* currently allocated buffers length */ 
    int buffer_pos;             /* end of data in buffer*/ 
    int still_running;          /* Is background url fetch still in progress */ 
};
 
typedef struct fcurl_data URL_FILE;
 
/* exported functions */ 
URL_FILE *url_fopen(const char *url,const char *operation);
int url_fclose(URL_FILE *file);
int url_feof(URL_FILE *file);
size_t url_url_fread(void *ptr, size_t size, size_t nmemb, URL_FILE *file);
char * url_fgets(char *ptr, int size, URL_FILE *file);
void url_rewind(URL_FILE *file);
 
/* we use a global one for convenience */ 
CURLM *multi_handle;
 
/* curl calls this routine to get more data */ 
static size_t
write_callback(char *buffer,
               size_t size,
               size_t nitems,
               void *userp)
{
    char *newbuff;
    int rembuff;
 
    URL_FILE *url = (URL_FILE *)userp;
    size *= nitems;
 
    rembuff=url->buffer_len - url->buffer_pos; /* remaining space in buffer */ 
 
    if(size > rembuff)
    {
        /* not enough space in buffer */ 
        newbuff=realloc(url->buffer,url->buffer_len + (size - rembuff));
        if(newbuff==NULL)
        {
            fprintf(stderr,"callback buffer grow failed\n");
            size=rembuff;
        }
        else
        {
            /* realloc suceeded increase buffer size*/ 
            url->buffer_len+=size - rembuff;
            url->buffer=newbuff;
 
            /*printf("Callback buffer grown to %d bytes\n",url->buffer_len);*/ 
        }
    }
 
    memcpy(&url->buffer[url->buffer_pos], buffer, size);
    url->buffer_pos += size;
 
    /*fprintf(stderr, "callback %d size bytes\n", size);*/ 
 
    return size;
}
 
/* use to attempt to fill the read buffer up to requested number of bytes */ 
static int
fill_buffer(URL_FILE *file,int want,int waittime)
{
    fd_set fdread;
    fd_set fdwrite;
    fd_set fdexcep;
    struct timeval timeout;
    int rc;
 
    /* only attempt to fill buffer if transactions still running and buffer
     * doesnt exceed required size already
     */ 
    if((!file->still_running) || (file->buffer_pos > want))
        return 0;
 
    /* attempt to fill buffer */ 
    do
    {
        int maxfd = -1;
        long curl_timeo = -1;
 
        FD_ZERO(&fdread);
        FD_ZERO(&fdwrite);
        FD_ZERO(&fdexcep);
 
        /* set a suitable timeout to fail on */ 
        timeout.tv_sec = 60; /* 1 minute */ 
        timeout.tv_usec = 0;
 
        curl_multi_timeout(multi_handle, &curl_timeo);
        if(curl_timeo >= 0) {
          timeout.tv_sec = curl_timeo / 1000;
          if(timeout.tv_sec > 1)
            timeout.tv_sec = 1;
          else
            timeout.tv_usec = (curl_timeo % 1000) * 1000;
        }
 
        /* get file descriptors from the transfers */ 
        curl_multi_fdset(multi_handle, &fdread, &fdwrite, &fdexcep, &maxfd);
 
        /* In a real-world program you OF COURSE check the return code of the
           function calls.  On success, the value of maxfd is guaranteed to be
           greater or equal than -1.  We call select(maxfd + 1, ...), specially
           in case of (maxfd == -1), we call select(0, ...), which is basically
           equal to sleep. */ 
 
        rc = select(maxfd+1, &fdread, &fdwrite, &fdexcep, &timeout);
 
        switch(rc) {
        case -1:
            /* select error */ 
            break;
 
        case 0:
            break;
 
        default:
            /* timeout or readable/writable sockets */ 
            /* note we *could* be more efficient and not wait for
             * CURLM_CALL_MULTI_PERFORM to clear here and check it on re-entry
             * but that gets messy */ 
            while(curl_multi_perform(multi_handle, &file->still_running) ==
                  CURLM_CALL_MULTI_PERFORM);
 
            break;
        }
    } while(file->still_running && (file->buffer_pos < want));
    return 1;
}
 
/* use to remove want bytes from the front of a files buffer */ 
static int
use_buffer(URL_FILE *file,int want)
{
    /* sort out buffer */ 
    if((file->buffer_pos - want) <=0)
    {
        /* ditch buffer - write will recreate */ 
        if(file->buffer)
            free(file->buffer);
 
        file->buffer=NULL;
        file->buffer_pos=0;
        file->buffer_len=0;
    }
    else
    {
        /* move rest down make it available for later */ 
        memmove(file->buffer,
                &file->buffer[want],
                (file->buffer_pos - want));
 
        file->buffer_pos -= want;
    }
    return 0;
}
 
 
 
URL_FILE *
url_fopen(const char *url,const char *operation)
{
    /* this code could check for URLs or types in the 'url' and
       basicly use the real fopen() for standard files */ 
 
    URL_FILE *file;
    (void)operation;
 
    file = malloc(sizeof(URL_FILE));
    if(!file)
        return NULL;
 
    memset(file, 0, sizeof(URL_FILE));
 
    if((file->handle.file=fopen(url,operation)))
    {
        file->type = CFTYPE_FILE; /* marked as URL */ 
    }
    else
    {
        file->type = CFTYPE_CURL; /* marked as URL */ 
        file->handle.curl = curl_easy_init();
 
        curl_easy_setopt(file->handle.curl, CURLOPT_URL, url);
        curl_easy_setopt(file->handle.curl, CURLOPT_WRITEDATA, file);
        curl_easy_setopt(file->handle.curl, CURLOPT_VERBOSE, 0L);
        curl_easy_setopt(file->handle.curl, CURLOPT_WRITEFUNCTION, write_callback);
 
        if(!multi_handle)
            multi_handle = curl_multi_init();
 
        curl_multi_add_handle(multi_handle, file->handle.curl);
 
        /* lets start the fetch */ 
        while(curl_multi_perform(multi_handle, &file->still_running) ==
              CURLM_CALL_MULTI_PERFORM );
 
        if((file->buffer_pos == 0) && (!file->still_running))
        {
            /* if still_running is 0 now, we should return NULL */ 
 
            /* make sure the easy handle is not in the multi handle anymore */ 
            curl_multi_remove_handle(multi_handle, file->handle.curl);
 
            /* cleanup */ 
            curl_easy_cleanup(file->handle.curl);
 
            free(file);
 
            file = NULL;
        }
    }
    return file;
}
 
int
url_fclose(URL_FILE *file)
{
    int ret=0;/* default is good return */ 
 
    switch(file->type)
    {
    case CFTYPE_FILE:
        ret=fclose(file->handle.file); /* passthrough */ 
        break;
 
    case CFTYPE_CURL:
        /* make sure the easy handle is not in the multi handle anymore */ 
        curl_multi_remove_handle(multi_handle, file->handle.curl);
 
        /* cleanup */ 
        curl_easy_cleanup(file->handle.curl);
        break;
 
    default: /* unknown or supported type - oh dear */ 
        ret=EOF;
        errno=EBADF;
        break;
 
    }
 
    if(file->buffer)
        free(file->buffer);/* free any allocated buffer space */ 
 
    free(file);
 
    return ret;
}
 
int
url_feof(URL_FILE *file)
{
    int ret=0;
 
    switch(file->type)
    {
    case CFTYPE_FILE:
        ret=feof(file->handle.file);
        break;
 
    case CFTYPE_CURL:
        if((file->buffer_pos == 0) && (!file->still_running))
            ret = 1;
        break;
    default: /* unknown or supported type - oh dear */ 
        ret=-1;
        errno=EBADF;
        break;
    }
    return ret;
}
 
size_t
url_url_fread(void *ptr, size_t size, size_t nmemb, URL_FILE *file)
{
    size_t want;
 
    switch(file->type)
    {
    case CFTYPE_FILE:
        want=url_fread(ptr,size,nmemb,file->handle.file);
        break;
 
    case CFTYPE_CURL:
        want = nmemb * size;
 
        fill_buffer(file,want,1);
 
        /* check if theres data in the buffer - if not fill_buffer()
         * either errored or EOF */ 
        if(!file->buffer_pos)
            return 0;
 
        /* ensure only available data is considered */ 
        if(file->buffer_pos < want)
            want = file->buffer_pos;
 
        /* xfer data to caller */ 
        memcpy(ptr, file->buffer, want);
 
        use_buffer(file,want);
 
        want = want / size;     /* number of items - nb correct op - checked
                                 * with glibc code*/ 
 
        /*printf("(url_fread) return %d bytes %d left\n", want,file->buffer_pos);*/ 
        break;
 
    default: /* unknown or supported type - oh dear */ 
        want=0;
        errno=EBADF;
        break;
 
    }
    return want;
}
 
char *
url_fgets(char *ptr, int size, URL_FILE *file)
{
    int want = size - 1;/* always need to leave room for zero termination */ 
    int loop;
 
    switch(file->type)
    {
    case CFTYPE_FILE:
        ptr = fgets(ptr,size,file->handle.file);
        break;
 
    case CFTYPE_CURL:
        fill_buffer(file,want,1);
 
        /* check if theres data in the buffer - if not fill either errored or
         * EOF */ 
        if(!file->buffer_pos)
            return NULL;
 
        /* ensure only available data is considered */ 
        if(file->buffer_pos < want)
            want = file->buffer_pos;
 
        /*buffer contains data */ 
        /* look for newline or eof */ 
        for(loop=0;loop < want;loop++)
        {
            if(file->buffer[loop] == '\n')
            {
                want=loop+1;/* include newline */ 
                break;
            }
        }
 
        /* xfer data to caller */ 
        memcpy(ptr, file->buffer, want);
        ptr[want]=0;/* allways null terminate */ 
 
        use_buffer(file,want);
 
        /*printf("(fgets) return %d bytes %d left\n", want,file->buffer_pos);*/ 
        break;
 
    default: /* unknown or supported type - oh dear */ 
        ptr=NULL;
        errno=EBADF;
        break;
    }
 
    return ptr;/*success */ 
}
 
void
url_rewind(URL_FILE *file)
{
    switch(file->type)
    {
    case CFTYPE_FILE:
        rewind(file->handle.file); /* passthrough */ 
        break;
 
    case CFTYPE_CURL:
        /* halt transaction */ 
        curl_multi_remove_handle(multi_handle, file->handle.curl);
 
        /* restart */ 
        curl_multi_add_handle(multi_handle, file->handle.curl);
 
        /* ditch buffer - write will recreate - resets stream pos*/ 
        if(file->buffer)
            free(file->buffer);
 
        file->buffer=NULL;
        file->buffer_pos=0;
        file->buffer_len=0;
 
        break;
 
    default: /* unknown or supported type - oh dear */ 
        break;
 
    }
 
}


BgzfData::BgzfData(void)
    : UncompressedBlockSize(DEFAULT_BLOCK_SIZE)
    , CompressedBlockSize(MAX_BLOCK_SIZE)
    , BlockLength(0)
    , BlockOffset(0)
    , BlockAddress(0)
    , IsOpen(false)
    , IsWriteOnly(false)
    , Stream(NULL)
    , UncompressedBlock(NULL)
    , CompressedBlock(NULL)
{
    try {
        CompressedBlock   = new char[CompressedBlockSize];
        UncompressedBlock = new char[UncompressedBlockSize];
    } catch( std::bad_alloc& ba ) {
        printf("BGZF ERROR: unable to allocate memory for our BGZF object.\n");
        exit(1);
    }
}

// destructor
BgzfData::~BgzfData(void) {
    if( CompressedBlock )   { delete[] CompressedBlock;   }
    if( UncompressedBlock ) { delete[] UncompressedBlock; }
}

// closes BGZF file
void BgzfData::Close(void) {

	// skip if file not open, otherwise set flag
    if ( !IsOpen ) return;
    IsOpen = false;

    // flush the current BGZF block
    if ( IsWriteOnly ) FlushBlock();

    // write an empty block (as EOF marker)
    int blockLength = DeflateBlock();
    fwrite(CompressedBlock, 1, blockLength, Stream);
    
    // flush and close
    fflush(Stream);
    fclose(Stream);
}

// compresses the current block
int BgzfData::DeflateBlock(void) {

    // initialize the gzip header
    char* buffer = CompressedBlock;
    memset(buffer, 0, 18);
    buffer[0]  = GZIP_ID1;
    buffer[1]  = (char)GZIP_ID2;
    buffer[2]  = CM_DEFLATE;
    buffer[3]  = FLG_FEXTRA;
    buffer[9]  = (char)OS_UNKNOWN;
    buffer[10] = BGZF_XLEN;
    buffer[12] = BGZF_ID1;
    buffer[13] = BGZF_ID2;
    buffer[14] = BGZF_LEN;

    // loop to retry for blocks that do not compress enough
    int inputLength = BlockOffset;
    int compressedLength = 0;
    unsigned int bufferSize = CompressedBlockSize;

    while(true) {
        
        // initialize zstream values
        z_stream zs;
        zs.zalloc    = NULL;
        zs.zfree     = NULL;
        zs.next_in   = (Bytef*)UncompressedBlock;
        zs.avail_in  = inputLength;
        zs.next_out  = (Bytef*)&buffer[BLOCK_HEADER_LENGTH];
        zs.avail_out = bufferSize - BLOCK_HEADER_LENGTH - BLOCK_FOOTER_LENGTH;

        // initialize the zlib compression algorithm
        if(deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, GZIP_WINDOW_BITS, Z_DEFAULT_MEM_LEVEL, Z_DEFAULT_STRATEGY) != Z_OK) {
            printf("BGZF ERROR: zlib deflate initialization failed.\n");
            exit(1);
        }

        // compress the data
        int status = deflate(&zs, Z_FINISH);
        if(status != Z_STREAM_END) {

            deflateEnd(&zs);

            // reduce the input length and try again
            if(status == Z_OK) {
                inputLength -= 1024;
                if(inputLength < 0) {
                    printf("BGZF ERROR: input reduction failed.\n");
                    exit(1);
                }
                continue;
            }

            printf("BGZF ERROR: zlib::deflateEnd() failed.\n");
            exit(1);
        }

        // finalize the compression routine
        if(deflateEnd(&zs) != Z_OK) {
            printf("BGZF ERROR: zlib::deflateEnd() failed.\n");
            exit(1);
        }

        compressedLength = zs.total_out;
        compressedLength += BLOCK_HEADER_LENGTH + BLOCK_FOOTER_LENGTH;
        if(compressedLength > MAX_BLOCK_SIZE) {
            printf("BGZF ERROR: deflate overflow.\n");
            exit(1);
        }

        break;
    }

    // store the compressed length
    BgzfData::PackUnsignedShort(&buffer[16], (unsigned short)(compressedLength - 1));

    // store the CRC32 checksum
    unsigned int crc = crc32(0, NULL, 0);
    crc = crc32(crc, (Bytef*)UncompressedBlock, inputLength);
    BgzfData::PackUnsignedInt(&buffer[compressedLength - 8], crc);
    BgzfData::PackUnsignedInt(&buffer[compressedLength - 4], inputLength);

    // ensure that we have less than a block of data left
    int remaining = BlockOffset - inputLength;
    if(remaining > 0) {
        if(remaining > inputLength) {
            printf("BGZF ERROR: after deflate, remainder too large.\n");
            exit(1);
        }
        memcpy(UncompressedBlock, UncompressedBlock + inputLength, remaining);
    }

    BlockOffset = remaining;
    return compressedLength;
}

// flushes the data in the BGZF block
void BgzfData::FlushBlock(void) {

    // flush all of the remaining blocks
    while(BlockOffset > 0) {

        // compress the data block
        int blockLength = DeflateBlock();

        // flush the data to our output stream
        int numBytesWritten = fwrite(CompressedBlock, 1, blockLength, Stream);

        if(numBytesWritten != blockLength) {
          printf("BGZF ERROR: expected to write %u bytes during flushing, but wrote %u bytes.\n", blockLength, numBytesWritten);
          exit(1);
        }
              
        BlockAddress += blockLength;
    }
}

// de-compresses the current block
int BgzfData::InflateBlock(const int& blockLength) {

    // Inflate the block in m_BGZF.CompressedBlock into m_BGZF.UncompressedBlock
    z_stream zs;
    zs.zalloc    = NULL;
    zs.zfree     = NULL;
    zs.next_in   = (Bytef*)CompressedBlock + 18;
    zs.avail_in  = blockLength - 16;
    zs.next_out  = (Bytef*)UncompressedBlock;
    zs.avail_out = UncompressedBlockSize;

    int status = inflateInit2(&zs, GZIP_WINDOW_BITS);
    if (status != Z_OK) {
        printf("BGZF ERROR: could not decompress block - zlib::inflateInit() failed\n");
        return -1;
    }

    status = inflate(&zs, Z_FINISH);
    if (status != Z_STREAM_END) {
        inflateEnd(&zs);
        printf("BGZF ERROR: could not decompress block - zlib::inflate() failed\n");
        return -1;
    }

    status = inflateEnd(&zs);
    if (status != Z_OK) {
        printf("BGZF ERROR: could not decompress block - zlib::inflateEnd() failed\n");
        return -1;
    }

    return zs.total_out;
}

// opens the BGZF file for reading (mode is either "rb" for reading, or "wb" for writing)
bool BgzfData::Open(const string& filename, const char* mode) {

	// determine open mode
    if ( strcmp(mode, "rb") == 0 ) {
        IsWriteOnly = false;
    } else if ( strcmp(mode, "wb") == 0) {
        IsWriteOnly = true;
    } else {
        printf("BGZF ERROR: unknown file mode: %s\n", mode);
        return false; 
    }

    // open Stream to read to/write from file, stdin, or stdout
    // stdin/stdout option contributed by Aaron Quinlan (2010-Jan-03)
    if ( (filename != "stdin") && (filename != "stdout") ) {
        // read/write BGZF data to/from a file
//         Stream = fopen64(filename.c_str(), mode);    
        //if ((strncmp(filename.c_str(), "http", 4) != 0) && (strncmp(filename.c_str(), "ftp", 3) != 0))
            Stream = url_fopen(filename.c_str(), mode);
        //else {
        //   printf("yep\n");
        //    OpenStream(filename);
        //}
    }
    else if ( (filename == "stdin") && (strcmp(mode, "rb") == 0 ) ) { 
        // read BGZF data from stdin
//         Stream = freopen64(NULL, mode, stdin);
        Stream = freopen(NULL, mode, stdin);
    }
    else if ( (filename == "stdout") && (strcmp(mode, "wb") == 0) ) { 
        // write BGZF data to stdout
//         Stream = freopen64(NULL, mode, stdout);
        Stream = freopen(NULL, mode, stdout);
    }

    if(!Stream) {
        printf("BGZF ERROR: unable to open file %s\n", filename.c_str() );
        return false;
    }
    
    // set flag, return success
    IsOpen = true;
    return true;
}

// opens a CURL stream for reading
bool BgzfData::OpenStream(const string& url) {
    CURL *curl;
    CURLcode res;
    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);  // in BGZF.h
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &Stream);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(Stream);
    }
    else {
        printf("CURL ERROR: unable to open remote file %s\n", url.c_str() );
        return false;        
    }
    // set flag, return success
    IsOpen = true;
    return true;
}


// reads BGZF data into a byte buffer
int BgzfData::Read(char* data, const unsigned int dataLength) {

   if (dataLength == 0) return 0;

   char* output = data;
   unsigned int numBytesRead = 0;
   while (numBytesRead < dataLength) {

       int bytesAvailable = BlockLength - BlockOffset;
       if ( bytesAvailable <= 0 ) {
           if (!ReadBlock()) return -1; 
           bytesAvailable = BlockLength - BlockOffset;
           if (bytesAvailable <= 0) break;
       }

       char* buffer   = UncompressedBlock;
       int copyLength = min( (int)(dataLength-numBytesRead), bytesAvailable );
       memcpy(output, buffer + BlockOffset, copyLength);

       BlockOffset  += copyLength;
       output       += copyLength;
       numBytesRead += copyLength;
   }

   if ( BlockOffset == BlockLength ) {
       BlockAddress = ftell64(Stream);
       BlockOffset  = 0;
       BlockLength  = 0;
   }

   return numBytesRead;
}

// reads a BGZF block
bool BgzfData::ReadBlock(void) {

    char    header[BLOCK_HEADER_LENGTH];
    int64_t blockAddress = ftell64(Stream);
    
    int count = url_fread(header, 1, sizeof(header), Stream);
    if (count == 0) {
        BlockLength = 0;
        return true;
    }

    if (count != sizeof(header)) {
        printf("BGZF ERROR: read block failed - could not read block header\n");
        return false;
    }

    if (!BgzfData::CheckBlockHeader(header)) {
        printf("BGZF ERROR: read block failed - invalid block header\n");
        return false;
    }

    int blockLength = BgzfData::UnpackUnsignedShort(&header[16]) + 1;
    char* compressedBlock = CompressedBlock;
    memcpy(compressedBlock, header, BLOCK_HEADER_LENGTH);
    int remaining = blockLength - BLOCK_HEADER_LENGTH;

    count = url_fread(&compressedBlock[BLOCK_HEADER_LENGTH], 1, remaining, Stream);
    if (count != remaining) {
        printf("BGZF ERROR: read block failed - could not read data from block\n");
        return false;
    }

    count = InflateBlock(blockLength);
    if (count < 0) { 
      printf("BGZF ERROR: read block failed - could not decompress block data\n");
      return false;
    }

    if ( BlockLength != 0 )
        BlockOffset = 0;

    BlockAddress = blockAddress;
    BlockLength  = count;
    return true;
}

// seek to position in BGZF file
bool BgzfData::Seek(int64_t position) {

    int     blockOffset  = (position & 0xFFFF);
    int64_t blockAddress = (position >> 16) & 0xFFFFFFFFFFFFLL;

    if (fseek64(Stream, blockAddress, SEEK_SET) != 0) {
        printf("BGZF ERROR: unable to seek in file\n");
        return false;
    }

    BlockLength  = 0;
    BlockAddress = blockAddress;
    BlockOffset  = blockOffset;
    return true;
}

// get file position in BGZF file
int64_t BgzfData::Tell(void) {
    return ( (BlockAddress << 16) | (BlockOffset & 0xFFFF) );
}

// writes the supplied data into the BGZF buffer
unsigned int BgzfData::Write(const char* data, const unsigned int dataLen) {

    // initialize
    unsigned int numBytesWritten = 0;
    const char* input = data;
    unsigned int blockLength = UncompressedBlockSize;

    // copy the data to the buffer
    while(numBytesWritten < dataLen) {
      
        unsigned int copyLength = min(blockLength - BlockOffset, dataLen - numBytesWritten);
        char* buffer = UncompressedBlock;
        memcpy(buffer + BlockOffset, input, copyLength);

        BlockOffset     += copyLength;
        input           += copyLength;
        numBytesWritten += copyLength;

        if(BlockOffset == blockLength)
            FlushBlock();
    }

    return numBytesWritten;
}
