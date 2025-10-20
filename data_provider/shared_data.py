from data_provider.shared_construct import VAE,vae_loss,manual_info_loss,wasserstein_distance
import torch.nn as nn
from torch import optim
import time
import torch
import numpy as np
from torch.utils import data as data_utils
import sys
from torch.utils.data import DataLoader
from fed.update import LocalDataset
def get_shared_data(train_datasets,args):
    criterion = nn.MSELoss()
    client_models_x = {}
    client_models_emb = {}
    client_losses = {}
    data_nums = {}
    
    original_dim = args.num_nodes
    lantent_dim = args.channel
    d_llm = args.d_llm

    vae_local_model_x = VAE(original_dim, lantent_dim,device=args.device).float().to(args.device) 
    vae_local_model_emb = VAE(d_llm, lantent_dim,device=args.device).float().to(args.device)

    vae_local_optimizer_x = optim.Adam(vae_local_model_x.parameters(), lr=0.001)
    vae_local_optimizer_emb = optim.Adam(vae_local_model_x.parameters(), lr=0.001)
    shared_dataset_start_time = time.time()
    for global_epoch in range(args.vae_train_epochs):
        if(global_epoch == 0):
            global_vae_model_x = vae_local_model_x
            global_vae_model_emb = vae_local_model_emb
        global_state_dict_x = global_vae_model_x.state_dict()
        global_state_dict_emb = global_vae_model_emb.state_dict()

        for client_id,client_dataset in train_datasets.items():
            data_loader = DataLoader(LocalDataset(client_dataset), batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)
            vae_local_model_x.train()
            vae_local_model_emb.train()
            vae_local_model_x.load_state_dict(global_state_dict_x)
            vae_local_model_emb.load_state_dict(global_state_dict_emb)
            alpha1 = 10
            train_loss = {}
            data_nums[client_id] = len(data_loader)
            for epoch in range(args.vae_local_epochs):
                for i, (batch_x, batch_y, emb) in enumerate(data_loader): 
                    vae_local_optimizer_x.zero_grad()
                    inputs_x = batch_x.float().to(args.device)
                    inputs_emb = emb.float().to(args.device)
                    inputs_emb = inputs_emb.squeeze(-1)       # [B, N, D]

                    outputs_x, z_mean_x, z_log_var_x = vae_local_model_x(inputs_x)
                    outputs_emb, z_mean_emb, z_log_var_emb = vae_local_model_emb(inputs_emb)
                    
                    vaeloss_x = vae_loss(vae_local_model_x, inputs_x, outputs_x, z_mean_x, z_log_var_x)
                    input_size_x = original_dim
                    distribute_loss_x = wasserstein_distance(inputs_x, input_size_x,outputs_x)
                    mutual_information_x = manual_info_loss(inputs_x, outputs_x)

                    vaeloss_emb = vae_loss(vae_local_model_emb, inputs_emb, outputs_emb, z_mean_emb, z_log_var_emb)
                    input_size_emb = d_llm
                    distribute_loss_emb = wasserstein_distance(inputs_emb, input_size_emb,outputs_emb)
                    mutual_information_emb = manual_info_loss(inputs_emb, outputs_emb)

                    loss_x = vaeloss_x + distribute_loss_x + mutual_information_x 
                    loss_x.backward()
                    vae_local_optimizer_x.step()

                    loss_emb = vaeloss_emb + distribute_loss_emb + mutual_information_emb 
                    loss_emb.backward()
                    vae_local_optimizer_emb.step()
                    train_loss[client_id] = loss_x.item() + loss_emb.item()

            client_losses[client_id] = np.average(train_loss[client_id])
            client_models_x[client_id] = vae_local_model_x.state_dict()
            client_models_emb[client_id] = vae_local_model_emb.state_dict()
            print(f'Shared data Epoch [{epoch + 1}/{args.vae_local_epochs}],{client_id}  Loss: {client_losses[client_id]:.4f}')
            
    #Client synthesizes time series   
    local_sequences_per_client = {}
    shared_sequences_x_list = []
    shared_sequences_emb_list = []
    num_samples = args.shared_size

    with torch.no_grad():
        for client_id, data_loader in train_datasets.items():
            latent_samples_x = torch.randn(num_samples,lantent_dim).to(args.device)
            latent_samples_emb = torch.randn(num_samples,lantent_dim).to(args.device)
            
            vae_local_model_x.load_state_dict(client_models_x[client_id])
            vae_local_model_emb.load_state_dict(client_models_emb[client_id])
            
            generated_sequences_x = vae_local_model_x.decoder(latent_samples_x).view(-1,original_dim)
            generated_sequences_emb = vae_local_model_emb.decoder(latent_samples_emb).view(-1,d_llm)
            
            generated_sequence_x = generated_sequences_x.cpu().detach().numpy()
            generated_sequence_emb = generated_sequences_emb.cpu().detach().numpy()
            
            local_sequences_per_client[client_id] = generated_sequence_x
            shared_sequences_x_list.append(generated_sequence_x)    
            shared_sequences_emb_list.append(generated_sequence_emb)    

        # shared_dataset_np = np.vstack(shared_sequences_list)
        shared_dataset_x_np = np.array(shared_sequences_x_list)
        shared_dataset_emb_np = np.array(shared_sequences_emb_list)

        # shared_dataset_np = shared_dataset_np.reshape(-1, num_samples, self.original_dim)
        shared_dataset_x_tensor = torch.tensor(shared_dataset_x_np, dtype=torch.float32)
        shared_dataset_emb_tensor = torch.tensor(shared_dataset_emb_np, dtype=torch.float32)

        class SharedPairDataset(data_utils.Dataset):
            def __init__(self, x_data, emb_data):
                self.x_data = x_data
                self.emb_data = emb_data

            def __len__(self):
                return len(self.x_data)

            def __getitem__(self, idx):
                return self.x_data[idx], self.emb_data[idx]  # 返回元组

        # 实例化 dataset 并创建 dataloader
        shared_dataset = SharedPairDataset(shared_dataset_x_tensor, shared_dataset_emb_tensor)
        shared_dataset_loader = data_utils.DataLoader(
            shared_dataset,
            batch_size=args.local_bs,
            shuffle=True
        )

    print(f"Shared data shape: x {shared_dataset_x_tensor.shape}, emb {shared_dataset_emb_tensor.shape}")
    # 示例输出: Shared data shape: x torch.Size([1000, 38]), emb torch.Size([1000, 64])

    total_size_bytes = shared_dataset_x_tensor.numel() * shared_dataset_x_tensor.element_size() + \
                    shared_dataset_emb_tensor.numel() * shared_dataset_emb_tensor.element_size()
    total_size_mb = total_size_bytes / (1024 ** 2)
    print(f"Total size of shared_dataset: {total_size_mb:.2f} MB")

    print('cost time:', time.time() - shared_dataset_start_time)
    torch.cuda.empty_cache()

    return shared_dataset_loader
    