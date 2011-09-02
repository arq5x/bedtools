#!/bin/bash

source files.sh
source ranges.sh

for D in $D_RANGE
do
	for Q in $Q_RANGE
	do
		for I in $GM_I_RANGE
		do
			RT=`$SEQ/i_bsearch $D $Q $GM_I_RANGE 1`
			echo "$D,$Q,$I,$RT"
		done
	done
done
		
