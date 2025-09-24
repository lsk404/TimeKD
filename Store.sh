#!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH

root_path="./data/Exchange/"
# data_paths=("ETTh1" "ETTh2" "ETTm1" "ETTm2")
data_paths=("exchange_rate")
divides=("train" "val" "test") 
output_len_values=(24 36 48)

# 其他参数
device="cuda:0"
num_nodes=8
input_len=96
model_path="./gpt2_model"
tokenizer_path="./gpt2_tokenizer"
d_model=768
l_layer=12

# 创建日志目录
mkdir -p ./log

## 生成所有命令的临时文件
commands_file=$(mktemp)
for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    for output_len in "${output_len_values[@]}"; do
      timestamp=$(date +"%Y%m%d_%H%M%S")
      log_file="./log/${timestamp}_${data_path}_${output_len}_${divide}.log"
      echo "/root/anaconda3/envs/timekd/bin/python store_emb.py \
      --root_path '$root_path' \
      --data_path '$data_path' \
      --divide '$divide' \
      --device '$device' \
      --num_nodes $num_nodes \
      --input_len $input_len \
      --output_len $output_len \
      --model_path '$model_path' \
      --tokenizer_path '$tokenizer_path' \
      --d_model $d_model \
      --l_layer $l_layer \
      > '$log_file' 2>&1" >> "$commands_file"
    done
  done
done
# 等待所有后台任务完成
parallel -j 3 --progress < "$commands_file"
rm "$commands_file"
echo "all tasks are finished!"