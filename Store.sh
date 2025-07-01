 #!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
root_path="$HOME/autodl-tmp/all_datasets/ETT-small/"
data_paths=(
  # "$HOME/autodl-tmp/all_datasets/ETT-small"
  # "$HOME/autodl-tmp/all_datasets/weather"
  # "$HOME/autodl-tmp/all_datasets/Heartbeat"
  # "$HOME/autodl-tmp/all_datasets/traffic"
  "ETTh1"
  # "ETTh2"
  # "ETTm1"
  # "ETTm2"
)

divides=("train" "val" "test") 
device="cuda:0"
num_nodes=7
input_len=96
# output_len_values=(24 36 48 96 192)
output_len_values=(36 48 96)
# model_name=("gpt2")
model_path="./gpt2_model"
tokenizer_path="./gpt2_tokenizer"
d_model=768
l_layer=12

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    for output_len in "${output_len_values[@]}"; do
      timestamp=$(date +"%Y%m%d_%H%M%S")
      log_file="./log/${timestamp}_${data_path}_${output_len}_${divide}.log"
      python store_emb.py \
        --root_path $root_path \
        --data_path $data_path \
        --divide $divide \
        --device $device \
        --num_nodes $num_nodes \
        --input_len $input_len \
        --output_len $output_len \
        --model_path $model_path \
        --tokenizer_path $tokenizer_path \
        --d_model $d_model \
        --l_layer $l_layer 2>&1 | tee $log_file
    done
  done
done