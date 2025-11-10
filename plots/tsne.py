import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.TimeKD import Dual
import argparse
import time
import torch
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from fed.sampling import Distribute_data

def parse_args():
    parser = argparse.ArgumentParser()
    #model
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--root_path", type=str, default="./data/", help="data root path")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="just a data Name:ETTm1,ETTh1,ETTm2,ETTh2,not a path")
    parser.add_argument("--channel", type=int, default=512, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--seed', type=int, default=2036, help='random seed')
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    #fed
    parser.add_argument('--num_users', type=int, default=10, help='client counts')
    #data
    parser.add_argument("--data_split",type=str,default="val", help="read data from vald dataset or Test dataset")
    #tsne
    parser.add_argument('--tsne_client_count',type=int,default=4)
    return parser.parse_args()


def seed_it(seed):  
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

def visualize_with_tsne(args,client_name,ts,prompt,ts_enc, prompt_enc, ts_out, prompt_out,last_ts_att, last_prompt_att):
        tsne = TSNE(n_components=2, random_state=123,perplexity=30)

        def plot_tsne(data, title, color, filename):
            data_flat = data.reshape(-1, data.shape[-1])
            results_tsne = tsne.fit_transform(data_flat.cpu().detach().numpy())
            # path = f"./Vis/{args.data_path}/"
            # os.makedirs(path, exist_ok=True)
            # np.save(f'{path}/{filename}_x.npy',results_tsne[:, 0])
            # np.save(f'{path}/{filename}_y.npy',results_tsne[:, 1])
            plt.figure(figsize=(6, 5))
            plt.scatter(results_tsne[:, 0], results_tsne[:, 1], c=color, label=title, alpha=0.5)

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)    
            # plt.legend()
            # plt.title(f't-SNE Visualization of {title}')

            path = f"./Results/tsne/Figs/{args.data_path}_{args.data_split}/{client_name}/"
            os.makedirs(path, exist_ok=True)
            plt.savefig(f"{path}{filename}.png")
            print(f"Saved Figure to {path}{filename}.png")
            plt.close()

        # print(ts.shape)
        # print(prompt.shape)
        # print(ts_enc.shape)
        # print(prompt_enc.shape)
        # print(ts_out.shape)
        # print(prompt_out.shape)
        # print(last_ts_att.shape)
        # print(last_prompt_att.shape)

        colors = plt.cm.tab10(range(8))
        if(ts is not None): plot_tsne(ts, 'Time Series', colors[0], 'Time Series')
        if(prompt is not None): plot_tsne(prompt, 'Prompt', colors[1], 'Prompt')
        if(ts_enc is not None): plot_tsne(ts_enc, 'Time Series Encoders', colors[2], 'Time Series Encoders')
        if(prompt_enc is not None): plot_tsne(prompt_enc, 'Prompt Encoders', colors[3], 'Prompt Encoders')
        if(ts_out is not None): plot_tsne(ts_out, 'Time Series Outputs', colors[4], 'Time Series Outputs')
        if(prompt_out is not None): plot_tsne(prompt_out, 'Prompt Outputs', colors[5], 'Prompt Outputs')
        if(last_ts_att is not None): plot_tsne(last_ts_att, 'Last Time Series Attentions', colors[6], 'Last Time Series Attentions')
        if(last_prompt_att is not None): plot_tsne(last_prompt_att, 'Last Prompt Attentions', colors[7], 'Last Prompt Attentionss')

def client_tsne(args,dataset,client_name="server",model_dir_path=None):
    
    seed_it(args.seed)
    device = torch.device(args.device)
    model = Dual(
            device=args.device, channel=args.channel, num_nodes=args.num_nodes, seq_len=args.seq_len, pred_len=args.pred_len, 
            dropout_n=args.dropout_n, d_llm=args.d_llm, e_layer=args.e_layer, head=args.head
        )
    if(model_dir_path is None):
        print("path Not set")
        return
    else:
        path = model_dir_path

    if(client_name.startswith("server")):
        model.load_state_dict(torch.load(path + "best_model.pth"), strict=False)
    else:
        model.load_state_dict(torch.load(path + f"{client_name}.pth"), strict=False)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=args.num_workers)

    if(args.data_split == "test"):
        for iter, (x, y) in enumerate(data_loader):
            testx = torch.Tensor(x).to(device).float()
            testy = torch.Tensor(y).to(device).float()
            with torch.no_grad():
                ts_enc, _, ts_out, _, _, _ = model(testx, None)
            if iter < 50:
                if(iter == 0):
                    tmp_ts = testx
                    tmp_ts_enc = ts_enc
                    tmp_ts_out = ts_out
                else:
                    tmp_ts = torch.cat([tmp_ts,testx],dim=0)
                    tmp_ts_enc = torch.cat([tmp_ts_enc,ts_enc],dim=0)
                    tmp_ts_out = torch.cat([tmp_ts_out,ts_out],dim=0)
            elif iter == 50:
                visualize_with_tsne(args,client_name,tmp_ts,None,tmp_ts_enc,None,tmp_ts_out,None,None,None)
            elif iter > 50:
                break
    elif(args.data_split == "val"):
        for iter, (x, y, emb) in enumerate(data_loader):
            valx = torch.Tensor(x).to(device).float()
            valy = torch.Tensor(y).to(device).float()
            emb = torch.Tensor(emb).to(device).float()
            while emb.dim() > 3 and emb.shape[-1] == 1:
                emb = emb.squeeze(-1)
            # print(valx.shape)
            # print(emb.shape)
            with torch.no_grad():
                ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = model(valx, emb)
                
            if iter < 50:
                if(iter == 0):
                    tmp_ts = valx
                    tmp_prompt = emb
                    tmp_ts_enc = ts_enc
                    tmp_ts_out = ts_out
                    tmp_prompt_enc = prompt_enc
                    tmp_prompt_out = prompt_out
                    tmp_ts_att = ts_att
                    tmp_prompt_att = prompt_att
                else:
                    tmp_ts = torch.cat([tmp_ts,valx],dim=0)
                    tmp_prompt = torch.cat([tmp_prompt,emb],dim=0)
                    tmp_ts_enc = torch.cat([tmp_ts_enc,ts_enc],dim=0)
                    tmp_ts_out = torch.cat([tmp_ts_out,ts_out],dim=0)
                    tmp_prompt_enc = torch.cat([tmp_prompt_enc,prompt_enc],dim=0)
                    tmp_prompt_out = torch.cat([tmp_prompt_out,prompt_out],dim=0)
                    tmp_ts_att = torch.cat([tmp_ts_att,ts_att],dim=0)
                    tmp_prompt_att = torch.cat([tmp_prompt_att,prompt_att],dim=0)
            elif iter == 50:
                visualize_with_tsne(args,client_name,tmp_ts,tmp_prompt,tmp_ts_enc,tmp_prompt_enc,tmp_ts_out,tmp_prompt_out,tmp_ts_att,tmp_prompt_att)
            elif iter > 50:
                break
if __name__ == "__main__":
    t1 = time.time()
    args = parse_args()
    print(args)
    model_dir_path = "./logs/2025-11-05-00:10:30-/ETTh1/24_48_2_0.0001_0.5_6666_0.01_fed/"
    print("model_dir:",model_dir_path)

    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
        }
    data_class = data_map.get(args.data_path, Dataset_Custom)
    if(args.data_split == "val"):
        val_set = data_class(flag='val', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path,root_path=args.root_path)
        client_dataset_dict = Distribute_data(val_set, args.num_users,method="contiguous")# 为用户分配数据
        for client_id in range(1,args.tsne_client_count+1):
            client_name = f"client_{client_id}"
            client_tsne(args,client_dataset_dict[client_name],client_name=client_name,model_dir_path=model_dir_path)
            client_tsne(args,client_dataset_dict[client_name],client_name=f"server_{client_id}",model_dir_path=model_dir_path)
    else:
        test_set = data_class(flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path,root_path=args.root_path)
        for client_id in range(1,args.tsne_client_count+1):
            client_name = f"client_{client_id}"
            client_tsne(args,test_set,client_name=client_name,model_dir_path=model_dir_path)
        client_tsne(args,test_set,client_name="server",model_dir_path=model_dir_path)
    
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
