import torch
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from clm import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./data/")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=10)
    return parser.parse_args()

def get_dataset(root_path,data_path, flag, input_len, output_len):
    # datasets = {
    #     'ETTh1': Dataset_ETT_hour,
    #     'ETTh2': Dataset_ETT_hour,
    #     'ETTm1': Dataset_ETT_minute,
    #     'ETTm2': Dataset_ETT_minute,
    # }
    if("ETTh1" in data_path or "ETTh2" in data_path) :
        dataset_class = Dataset_ETT_hour
    elif("ETTm1" in data_path or "ETTm2" in data_path):
        dataset_class = Dataset_ETT_minute
    else:
        dataset_class = Dataset_Custom
    # dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(root_path=root_path,flag=flag, size=[input_len, 0, output_len], data_path=data_path)

def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set = get_dataset(args.root_path,args.data_path, 'train', args.input_len, args.output_len)
    test_set = get_dataset(args.root_path,args.data_path, 'test', args.input_len, args.output_len)
    val_set = get_dataset(args.root_path,args.data_path, 'val', args.input_len, args.output_len)

    print(f"Length of train_hd_set: {len(train_set)}")  
    print(f"Length of test_hd_set: {len(test_set)}")
    print(f"Length of val_hd_set: {len(val_set)}")

    data_loader = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    }[args.divide]

    gen_prompt_emb = GenPromptEmb(
        data_path=args.data_path,
        model_name=args.model_name,
        num_nodes=args.num_nodes,
        device=args.device,
        input_len=args.input_len,
        output_len=args.output_len,
        d_model=args.d_model,
        l_layer=args.l_layers,
    ).to(device)

    print(args)

    save_path = f"{args.root_path}/{args.data_path}/{args.output_len}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"./Results/emb_time/"
    os.makedirs(emb_time_path, exist_ok=True)
    # max_token_counts = []

    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        embeddings = gen_prompt_emb.generate_embeddings(x.to(device), y.to(device), x_mark.to(device), y_mark.to(device))
        # max_token_counts.append(max_token_count)

        file_path = f"{save_path}{i}.h5"
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data=embeddings.detach().cpu().numpy())

        # Save and visualize the first sample
        # if i >= 0:
        #     break
    
    # avg_max_token_count = sum(max_token_counts) // len(max_token_counts)
    # print(f"Average max_token_count: {avg_max_token_count}")

    # max_max_token_count = max(max_token_counts)
    # print(f"Max max_token_count: {max_max_token_count}")

if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")
