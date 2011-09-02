#!/bin/bash

source files.sh
source ranges.sh

for D in $D_RANGE
do
	for Q in $Q_RANGE
	do
		for I in $GM_I_RANGE
		do
			RT=`$CUDA/b_search $D $Q 1 $DEVICE`
			echo "$D,$Q,$I,$RT"
		done
	done
done
		
