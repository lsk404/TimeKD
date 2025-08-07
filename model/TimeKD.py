import torch
import torch.nn as nn
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class Dual(nn.Module):
    def __init__(
        self,
        device="cuda:7",
        channel=768,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=768,
        e_layer=1,
        head=8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n = dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.head = head

        # Norm
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
       
        # Emb
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.token_to_feature = nn.Linear(self.d_llm, self.channel).to(self.device)

        # Time Series Encoder
        self.ts_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=1,attention_dropout=self.dropout_n,output_attention=True), d_model=self.channel, n_heads=self.head),
                    d_model=self.channel,
                    d_ff=4*self.d_llm,
                    dropout=self.dropout_n,
                    activation='relu'
                ) for l in range(self.e_layer)
            ],
            norm_layer=torch.nn.LayerNorm(self.channel)
        ).to(self.device)

        # Prompt Encoder
        self.prompt_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=1,attention_dropout=self.dropout_n,output_attention=True), d_model=self.channel, n_heads=self.head),
                    d_model=self.channel,
                    d_ff=4*self.d_llm,
                    dropout=self.dropout_n,
                    activation='relu'
                ) for l in range(self.e_layer)
            ],
            norm_layer=torch.nn.LayerNorm(self.channel)
        ).to(self.device)

        # Projection
        self.ts_proj = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
        self.prompt_proj = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, x, prompt_emb):
        if prompt_emb is not None:
            prompt_emb = prompt_emb.float().squeeze() # B, N, E

            # TS Input
            ts_data = x.float() # B L N
            
            # TS Norm
            ts_norm = self.normalize_layers(ts_data, 'norm')
            ts_norm = ts_norm.permute(0,2,1)
            # _, _, N = ts_norm.shape

            # TS Emb
            ts_emb = self.length_to_feature(ts_norm) # B N 
            # ts_emb = self.enc_embedding(ts_data, x_mark)

            # TS Encoder
            ts_enc, ts_att = self.ts_encoder(ts_emb)
            ts_att_last = ts_att[-1]
            ts_att_avg = ts_att_last.mean(dim=0)

            # Prompt Encoder
            # print(prompt_emb.shape)
            prompt_emb = self.token_to_feature(prompt_emb) # B N E -> B N C
            prompt_enc, prompt_att = self.prompt_encoder(prompt_emb)
            prompt_att_last = prompt_att[-1]
            prompt_att_avg = prompt_att_last.mean(dim=0)

            #  TS Proj
            ts_out = self.ts_proj(ts_enc)
            ts_out = ts_out.permute(0,2,1)
            ts_out = self.normalize_layers(ts_out, 'denorm')

            #  Prompt Proj
            prompt_out = self.prompt_proj(prompt_enc)
            prompt_out = prompt_out.permute(0,2,1)
        
        else:
            # TS Input
            ts_data = x.float() # B L N
            prompt_enc = None
            prompt_out = None
            ts_att_avg = None
            prompt_att_avg = None
            
            # TS Norm
            ts_norm = self.normalize_layers(ts_data, 'norm')
            ts_norm = ts_norm.permute(0,2,1)
           
            # TS Emb
            ts_emb = self.length_to_feature(ts_norm) # B N C

            # TS Encoder
            ts_enc,_ = self.ts_encoder(ts_emb)

            # Proj
            ts_out = self.ts_proj(ts_enc)
            ts_out = ts_out.permute(0,2,1)
            ts_out = self.normalize_layers(ts_out, 'denorm')

        return ts_enc, prompt_enc, ts_out, prompt_out, ts_att_avg, prompt_att_avg
