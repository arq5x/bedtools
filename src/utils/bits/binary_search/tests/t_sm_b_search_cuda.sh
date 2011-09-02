#!/bin/bash

source files.sh
source ranges.sh

for D in $D_RANGE
do
	for Q in $Q_RANGE
	do
		for I in $SM_I_RANGE
		do
			RT=`$CUDA/t_sm_b_search $D $Q $I 1 $DEVICE`
			echo "$D,$Q,$I,$RT"
		done
	done
done
		
