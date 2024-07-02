for i in 1 2 3 4 5 6 7 8
do
CUDA_VISIBLE_DEVICES=3 python tools/train.py qfdet_configs/all.py --extra all-plus
done


