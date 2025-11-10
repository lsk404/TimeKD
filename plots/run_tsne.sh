#!/bin/bash
log_path="./Results/tsne/logs/"
mkdir -p "$log_path"

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_path}tsne_${timestamp}.log"

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0 

python ./plots/tsne.py \
  --root_path "./data/ETT-small" \
  --data_path "ETTh1" \
  --device cuda:0 \
  --num_nodes 7 \
  --seq_len 96 \
  --pred_len 24 \
  --seed 6666 \
  --channel 48 \
  --head 4 \
  --e_layer 2\
  --dropout_n 0.5\
  --data_split 'val'\
  --d_llm 768 2>&1 | tee -a "$log_file"