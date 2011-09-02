RANGE="5M_1M"
DEVICE_NAME="IntelCore2X9770"
#DEVICE_NAME="IntelXeonX5550"
#DEVICE_NAME="IntelXeonE5504"

POST="$RANGE.$DEVICE_NAME"
./b_search_seq.sh > bsearch_seq-$POST.txt
./sort_b_search_seq.sh > sort_bsearch_seq-$POST.txt
./i_search_seq.sh > i_bsearch_seq-$POST.txt
./t_search_seq.sh > t_bsearch_seq-$POST.txt
