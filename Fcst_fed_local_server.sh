#!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0 
seq_lens=(96)
pred_lens=(24)
d_llm=(768)
learning_rates=(1e-4)
channels=(32)
e_layers=(2)
dropout_ns=(0.5)
batch_sizes=(16)
temperatures=(0.7)
# model_name="gpt2"
model_path="./gpt2_model"
tokenizer_path="./gpt2_tokenizer"
root_path="./data/ETT-small/"
# data_paths=("ETTh1" "ETTh2" "ETTm1" "ETTm2")
data_paths=("ETTh1")
alphas=(0.90 0.95 0.99 1.0)
epochs=(300)
commands_file=$(mktemp)
for seq_len in "${seq_lens[@]}"; do 
  for pred_len in "${pred_lens[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for channel in "${channels[@]}"; do
        for dropout_n in "${dropout_ns[@]}"; do
          for e_layer in "${e_layers[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
              for temperature in "${temperatures[@]}"; do
                for data_path in "${data_paths[@]}"; do
                  for alpha in "${alphas[@]}"; do
                  log_path="./Results/Fcst/${data_path}/"
                  mkdir -p $log_path
                  timestamp=$(date +"%Y%m%d_%H%M%S")
                  log_file="${log_path}${timestamp}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dn${dropout_n}_bs${batch_size}_e${epochs}_temp${temperature}_alpha${alpha}.log"
cat >> "$commands_file" << EOF
python train_fed_local_server.py \
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
  --num_users 10\
  --frac 1.0 \
  --temperature $temperature \
  --local_ep 5\
  --alpha $alpha\
  --d_llm $d_llm 2>&1 | tee -a $log_file
EOF
                    done
                  done
                done
            done
          done
        done
      done
    done
  done
done


parallel -j 4 --progress < "$commands_file"
rm "$commands_file"
echo "All tasks are finished!"