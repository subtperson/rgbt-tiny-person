for i in 1 2 3 4 5 6 7 8 9 10
do
CUDA_VISIBLE_DEVICES=0 python tools/train.py qfdet_configs/all.py --extra all-plus
done