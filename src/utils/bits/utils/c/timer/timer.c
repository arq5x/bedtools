#include <sys/time.h>
#include "timer.h"

static struct timeval _start, _stop;

void start()
{
	gettimeofday(&_start,0);
}

void stop()
{
	gettimeofday(&_stop,0);
}

unsigned long report()
{
	return (_stop.tv_sec - _start.tv_sec) * 1000000 +  //seconds to microseconds
		_stop.tv_usec - _start.tv_usec;
}
