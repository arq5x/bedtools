#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "mt.h"


int main(void)
{


	/*
	int i, stime;
	long ltime;
	ltime = time(NULL);
	stime = (unsigned) ltime/2;
	srand(stime);
	*/

    int i;
    //unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
    //unsigned long init[4]={
		////(unsigned) time(NULL)/2, 	
		//(unsigned) time(NULL)/2, 	
		//(unsigned) time(NULL)/2, 	
		//(unsigned) time(NULL)/2
		//}, length=4;
    //init_by_array(init, length);
	init_genrand((unsigned) gettimeofday(NULL));
    printf("1000 outputs of genrand_int32()\n");
    for (i=0; i<1000; i++) {
      printf("%10lu ", genrand_int32());
      if (i%5==4) printf("\n");
    }

	/*
    printf("\n1000 outputs of genrand_real2()\n");
    for (i=0; i<1000; i++) {
      printf("%10.8f ", genrand_real2());
      if (i%5==4) printf("\n");
    }
    return 0;
	*/
}
