RANGE="10K-100K_1K-1M"
#RANGE="5M_1M"
DEVICE_NAME="GTX285"
POST="$RANGE.$DEVICE_NAME"
export DEVICE=0
./b_search_cuda.sh > b_search_cuda-$POST.txt
./sort_b_search_cuda.sh > sort_b_search_cuda-$POST.txt
./i_gm_b_search_cuda.sh > i_gm_b_search_cuda-$POST.txt
./i_sm_b_search_cuda.sh > i_sm_b_search_cuda-$POST.txt
./t_gm_b_search_cuda.sh > t_gm_b_search_cuda-$POST.txt
./t_sm_b_search_cuda.sh > t_sm_b_search_cuda-$POST.txt
./sort_i_gm_b_search_cuda.sh > sort_i_gm_b_search_cuda-$POST.txt
./sort_t_gm_b_search_cuda.sh > sort_i_tm_b_search_cuda-$POST.txt

DEVICE_NAME="9800GT"
POST="$RANGE.$DEVICE_NAME"
export DEVICE=1
./b_search_cuda.sh > b_search_cuda-$POST.txt
./sort_b_search_cuda.sh > sort_b_search_cuda-$POST.txt
./i_gm_b_search_cuda.sh > i_gm_b_search_cuda-$POST.txt
./i_sm_b_search_cuda.sh > i_sm_b_search_cuda-$POST.txt
./t_gm_b_search_cuda.sh > t_gm_b_search_cuda-$POST.txt
./t_sm_b_search_cuda.sh > t_sm_b_search_cuda-$POST.txt
./sort_i_gm_b_search_cuda.sh > sort_i_gm_b_search_cuda-$POST.txt
./sort_t_gm_b_search_cuda.sh > sort_i_tm_b_search_cuda-$POST.txt
