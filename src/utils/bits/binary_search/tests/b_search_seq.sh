#!/bin/bash

source files.sh
source ranges.sh

for D in $D_RANGE
do
	for Q in $Q_RANGE
	do
		for I in $GM_I_RANGE
		do
			RT=`$SEQ/bsearch $D $Q 1`
			echo "$D,$Q,$I,$RT"
		done
	done
done
		
