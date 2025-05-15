import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from model.TimeKD import Dual
from utils.kd_loss import KDLoss
from utils.metrics import MSE, MAE, metric
import faulthandler
faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:6", help="")
    parser.add_argument("--root_path", type=str, default="./data/", help="data root path")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data path")
    parser.add_argument("--channel", type=int, default=512, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lrate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--feature_w", type=float, default=0.01, help="weight of feature kd loss")
    parser.add_argument("--fcst_w", type=float, default=1, help="weight of forecast loss")
    parser.add_argument("--recon_w", type=float, default=0.5, help="weight of reconstruction loss")
    parser.add_argument("--att_w", type=float, default=0.01, help="weight of attention kd loss")
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument("--epochs", type=int, default=100, help="")
    parser.add_argument('--seed', type=int, default=2036, help='random seed')
    parser.add_argument(
        "--es_patience",
        type=int,
        default=50,
        help="quit if no improvement after this many iterations",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
        help="save path",
    )
    return parser.parse_args()


class trainer:
    def __init__(
        self,
        scaler,
        channel,
        num_nodes,
        seq_len,
        pred_len,
        dropout_n,
        d_llm,
        e_layer,
        head,
        lrate,
        wdecay,
        feature_w,
        fcst_w,
        recon_w,
        att_w,
        device,
        epochs
    ):
        self.model = Dual(
            device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len, 
            dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, head=head
        )
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 100), eta_min=1e-8, verbose=True)
        self.MSE = MSE
        self.MAE = MAE
        self.clip = 5
        self.scaler = scaler
        self.device = device

        self.feature_loss = 'smooth_l1'  
        self.fcst_loss = 'smooth_l1'
        self.recon_loss = 'smooth_l1'
        self.att_loss = 'smooth_l1'   
        self.fcst_w = 1
        self.recon_w = 0.5
        self.feature_w = 0.1     
        self.att_w = 0.01
        self.criterion = KDLoss(self.feature_loss, self.fcst_loss, self.recon_loss, self.att_loss,  self.feature_w,  self.fcst_w,  self.recon_w,  self.att_w)

        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))
        print(self.model)

    def train(self, x, y, emb):
        self.model.train()
        self.optimizer.zero_grad()
        ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(x, emb)
        loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip) 
        self.optimizer.step() 
        mse = self.MSE(ts_out, y) 
        mae = self.MAE(ts_out, y)
        return loss.item(), mse.item(), mae.item()

    def eval(self, x, y, emb):
        self.model.eval()
        with torch.no_grad():
            ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(x, emb)
            loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
            mse = self.MSE(ts_out, y)
            mae = self.MAE(ts_out, y)
        return loss.item(), mse.item(), mae.item()


def load_data(args):
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
        }
    # data_class = data_map.get(args.data_path, Dataset_Custom)
    if("ETTh1" in args.data_path or "ETTh2" in args.data_path):
        data_class = Dataset_ETT_hour
    elif ("ETTm1" in args.data_path or "ETTm2" in args.data_path):
        data_class = Dataset_ETT_minute
    else :
        data_class = Dataset_Custom
    train_set = data_class(flag='train', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path,root_path=args.root_path)
    val_set = data_class(flag='val', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path,root_path=args.root_path)
    test_set = data_class(flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path,root_path=args.root_path)
    
    scaler = train_set.scaler

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader, scaler

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


def main():
    args = parse_args()
    train_loader, val_loader, test_loader,scaler = load_data(args)
    seed_it(args.seed)
    device = torch.device(args.device)
    
    loss = 9999999
    test_log = 999999
    epochs_since_best_mse = 0

    path = os.path.join(args.save, args.data_path, 
                        f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.lrate}_{args.dropout_n}_{args.seed}_{args.att_w}/")
    if not os.path.exists(path):
        os.makedirs(path)
     
    his_loss = []
    val_time = []
    train_time = []
    test_time = []
    print(args)

    engine = trainer(
        scaler=scaler,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        head=args.head,
        lrate=args.lrate,
        wdecay=args.weight_decay,
        feature_w=args.feature_w,
        fcst_w=args.fcst_w,
        recon_w=args.recon_w,
        att_w=args.att_w,
        device=device,
        epochs=args.epochs
        )

    print("Start training...", flush=True)

    for i in range(1, args.epochs + 1):
        t1 = time.time()
        train_loss = []
        train_mse = []
        train_mae = []
        
        for iter, (x, y, emb) in enumerate(train_loader):
            trainx = torch.Tensor(x).to(device).float()
            trainy = torch.Tensor(y).to(device).float()
            emb = torch.Tensor(emb).to(device).float()
            metrics = engine.train(trainx, trainy, emb)
            train_loss.append(metrics[0])
            train_mse.append(metrics[1])
            train_mae.append(metrics[2])

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # Validation
        val_loss = []
        val_mse = []
        val_mae = []
        s1 = time.time()

        for iter, (x, y, emb) in enumerate(val_loader):
            valx = torch.Tensor(x).to(device).float()
            valy = torch.Tensor(y).to(device).float()
            emb = torch.Tensor(emb).to(device).float()
            metrics = engine.eval(valx, valy, emb)
            val_loss.append(metrics[0])
            val_mse.append(metrics[1])
            val_mae.append(metrics[2])

        s2 = time.time()
        log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mse = np.mean(train_mse)
        mtrain_mae = np.mean(train_mae)
        mvalid_loss = np.mean(val_loss)
        mvalid_mse = np.mean(val_mse)
        mvalid_mae = np.mean(val_mae)

        his_loss.append(mvalid_loss)
        print("-----------------------")

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train MSE: {:.4f}, Train MAE: {:.4f}"
        print(
            log.format(i, mtrain_loss, mtrain_mse, mtrain_mae),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid MSE: {:.4f}, Valid MAE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_mse, mvalid_mae),
            flush=True,
        )

        if mvalid_loss < loss:
            print("###Update tasks appear###")
            if i <= 10:
                # It is not necessary to print the results of the testset when epoch is less than n, because the model has not yet converged.
                loss = mvalid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mse = 0
                print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
                print("epoch: ", i)
            else:
                test_outputs = []
                test_y = []

                for iter, (x, y) in enumerate(test_loader):
                    testx = torch.Tensor(x).to(device).float()
                    testy = torch.Tensor(y).to(device).float()
                    with torch.no_grad():
                        preds = engine.model(testx, None)
                    test_outputs.append(preds[2])
                    test_y.append(testy)

                test_pre = torch.cat(test_outputs, dim=0)
                test_real = torch.cat(test_y, dim=0)

                amse = []
                amae = []
                
                for j in range(args.pred_len):
                    pred = test_pre[:, j,].to(device)
                    real = test_real[:, j, ].to(device)
                    errors = metric(pred, real)
                    # log = "Evaluate best model on test data for horizon {:d}, Test MSE: {:.4f}, Test MAE: {:.4f}"
                    amse.append(errors[0])
                    amae.append(errors[1])

                # log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
                # print(
                #     log.format(
                #         np.mean(amse), np.mean(amae)
                #     )
                # )

                if np.mean(amse) < test_log:
                    test_log = np.mean(amse)
                    loss = mvalid_loss
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mse = 0
                    print("Test low! Updating! Test MSE: {:.4f}, Test MAE: {:.4f}".format(np.mean(amse), np.mean(amae)), end=", ")
                    print("Test low! Updating! Valid Loss: {:.4f}".format(mvalid_loss), end=", ")

                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mse += 1
                    print("No update")

        else:
            epochs_since_best_mse += 1
            print("No update")

        engine.scheduler.step()

        if epochs_since_best_mse >= args.es_patience and i >= args.epochs//2: # early stop
            break

    # Output consumption
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))

    # Test
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))
   
    engine.model.load_state_dict(torch.load(path + "best_model.pth"), strict=False)
    
    test_outputs = []
    test_y = []
    
    test_start_time = time.time()
    for iter, (x, y) in enumerate(test_loader):
        testx = torch.Tensor(x).to(device).float()
        testy = torch.Tensor(y).to(device).float()
        with torch.no_grad():
            preds = engine.model(testx, None)
        test_outputs.append(preds[2])
        test_y.append(testy)


    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.cat(test_y, dim=0)

    amse = []
    amae = []
    
    for j in range(args.pred_len):
        pred = test_pre[:, j,].to(device)
        real = test_real[:, j, ].to(device)
        errors = metric(pred, real)
        amse.append(errors[0])
        amae.append(errors[1])
    

    test_end_time = time.time()
    print(f"Test time (total): {test_end_time - test_start_time:.4f} seconds")

    log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
    print(log.format(np.mean(amse), np.mean(amae)))
    print("Average Testing Time: {:.4f} secs".format(np.mean(test_time)))

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
