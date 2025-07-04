#!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

seq_lens=(96)
pred_lens=(24)
learning_rates=(1e-4)
channels=(64)
d_llm=(768)
e_layers=(2)
dropout_ns=(0.5)
batch_sizes=(16)
# model_name="gpt2"
model_path="gpt2"
tokenizer_path="gpt2"
root_path="/remote-home/data/ETT-small/"
data_path="ETTh1"
epochs=(100)


for seq_len in "${seq_lens[@]}"; do 
  for pred_len in "${pred_lens[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for channel in "${channels[@]}"; do
        for dropout_n in "${dropout_ns[@]}"; do
          for e_layer in "${e_layers[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
              log_path="./Results/Fcst/${data_path}/"
              mkdir -p $log_path
              timestamp=$(date +"%Y%m%d_%H%M%S")
              log_file="${log_path}${timestamp}_i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dn${dropout_n}_bs${batch_size}_e${epochs}_fed.log"
              python train.py \
                --root_path $root_path \
                --data_path $data_path \
                --device cuda:0 \
                --batch_size $batch_size \
                --num_nodes 7 \
                --seq_len $seq_len \
                --pred_len $pred_len \
                --epochs $epochs \
                --seed 6666 \
                --channel $channel \
                --head 4 \
                --lrate $learning_rate \
                --dropout_n $dropout_n \
                --e_layer $e_layer\
                --model_path $model_path \
                --tokenizer_path $tokenizer_path \
                --num_workers 10 \
                --d_llm $d_llm 2>&1 | tee -a $log_file
            done
          done
        done
      done
    done
  done
done
