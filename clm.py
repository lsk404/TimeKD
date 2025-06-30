import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from layers.Sub_CA import SCA
import re
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class MSK(nn.Module):
    def __init__(self, device="cuda:0", l_layer=6,model_path='gpt2'):
        super(MSK, self).__init__()
        self.device = device
        self.gpt2 = GPT2Model.from_pretrained(model_path, attn_implementation="eager",
                                              output_attentions=True, output_hidden_states=True)  #attn_implementation="sdpa" OR "eager"
        
        self.gpt2.h = self.gpt2.h[:l_layer]
        for param in self.gpt2.h.parameters():
            param.requires_grad = False

    def custom_forward(self,
                    input_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                    attention_mask: Optional[torch.FloatTensor] = None,
                    token_type_ids: Optional[torch.LongTensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    head_mask: Optional[torch.FloatTensor] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    encoder_attention_mask: Optional[torch.FloatTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    calibrated_mask: Optional[torch.FloatTensor] = None, 
                    ) -> Union[Tuple, dict]:

        output_attentions = output_attentions if output_attentions is not None else self.gpt2.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.gpt2.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.gpt2.config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt2.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.gpt2.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.gpt2.wte(inputs_embeds)
        position_embeds = self.gpt2.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        presents = () if use_cache else None

        for i, (block, layer_past) in enumerate(zip(self.gpt2.h, past_key_values)):
            attention_mask =  calibrated_mask
            
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]

            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
        
        hidden_states = self.gpt2.ln_f(hidden_states)
        hidden_states = hidden_states.view((-1,) + input_shape[1:] + (hidden_states.size(-1),))

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )

    def forward(self, x, calibrated_mask):
        calibrated_mask = calibrated_mask.to(self.device).float()
        calibrated_mask = calibrated_mask.unsqueeze(0)
        num_heads =  self.gpt2.config.n_head
        calibrated_mask = calibrated_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        output = self.custom_forward(
            inputs_embeds=x,
            calibrated_mask=calibrated_mask
        ).last_hidden_state

        return output

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path='ETTh1',
        model_path="gpt2",
        tokenizer_path="gpt2",
        num_nodes=7,
        device='cuda:6',
        input_len=96,
        output_len=96,
        d_model=768,
        l_layer=6
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.num_nodes = num_nodes
        self.device = device
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.l_layer = l_layer
        
        self.len = self.input_len - 1
        self.out_len = self.output_len -1

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.gpt2 = MSK(device=self.device, l_layer=self.l_layer,model_path=model_path)
        self.sub_ac = SCA(
            d_model=self.num_nodes, n_heads=1, d_ff=4*d_model, norm='LayerNorm',
            attn_dropout=0.1, dropout=0.1, pre_norm=True, activation="gelu",
            res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)
        for param in self.sub_ac.parameters():
            param.requires_grad = False


    def _generate_prompt(self, template_type):
        templates = {
            'GT_HD': {
                'ETTh1': "From <t1> to <t2>, the values were <value1, ..., valuen> every hour. The values for the next <output_len> hours are <gt1, ..., gtn>",
                'ETTh2': "From <t1> to <t2>, the values were <value1, ..., valuen> every hour. The values for the next <output_len> hours are <gt1, ..., gtn>",
                'Exchange': "From <t1> to <t2>, the exchange rates were <value1, ..., valuen> every day. The rates for the next <output_len> days are <gt1, ..., gtn>",
                'ETTm1': "From <t1> to <t2>, the values were <value1, ..., valuen> every 15 minutes. The values for the next <output_len> minutes are <gt1, ..., gtn>",
                'ETTm2': "From <t1> to <t2>, the values were <value1, ..., valuen> every 15 minutes. The values for the next <output_len> minutes are <gt1, ..., gtn>",
                'Weather': "From <t1> to <t2>, the values were <value1, ..., valuen> every 10 minutes. The values for the next <output_len> minutes are <gt1, ..., gtn>"
            },
            'HD': {
                'ETTh1': "From <t1> to <t2>, the values were <value1, ..., valuen> every hour. Forecast the values for the next <output_len> hours",
                'ETTh2': "From <t1> to <t2>, the values were <value1, ..., valuen> every hour. Forecast the values for the next <output_len> hours",
                'Exchange': "From <t1> to <t2>, the exchange rates were <value1, ..., valuen> every day. Forecast the rates for the next <output_len> days",
                'ETTm1': "From <t1> to <t2>, the values were <value1, ..., valuen> every 15 minutes. Forecast the values for the next <output_len> minutes",
                'ETTm2': "From <t1> to <t2>, the values were <value1, ..., valuen> every 15 minutes. Forecast the values for the next <output_len> minutes",
                'Weather': "From <t1> to <t2>, the values were <value1, ..., valuen> every 10 minutes. Forecast the values for the next <output_len> minutes"
            }
        }
        return templates[template_type].get(self.data_path, templates[template_type]['ETTh1'])
    

    def _generate_mask(self, token_types, max_length):
        mask = torch.zeros((max_length, max_length), device=self.device)

        language_indices = [i for i, t in enumerate(token_types) if t == "language"]
        time_series_indices = [i for i, t in enumerate(token_types) if t == "time_series"]

        language_indices = [i for i in language_indices if i < max_length]
        time_series_indices = [j for j in time_series_indices if j < max_length]

        for i in language_indices:
            for j in time_series_indices:
                mask[i, j] = -100  
                mask[j, i] = -100  

        return mask

    def _prepare_prompt(self, GT_HD, HD, x, y, x_mark, y_mark, i, j):
        values = x[i, :, j].flatten().tolist()
        values_str = ", ".join([str(int(value)) for value in values])

        gt_values = y[i, :, j].flatten().tolist()
        gt_str = ", ".join([str(int(value)) for value in gt_values])
    
        GT_HD_prompt = GT_HD.replace("value1, ..., valuen", values_str)
        HD_prompt = HD.replace("value1, ..., valuen", values_str)
        HD_prompt = HD_prompt.replace("output_len", str(self.output_len))
        GT_HD_prompt = GT_HD_prompt.replace("gt1, ..., gtn", gt_str)
        GT_HD_prompt = GT_HD_prompt.replace("output_len", str(self.output_len))

        if self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            hd_start_date = f"{int(x_mark[i, 0, 2]):02d}/{int(x_mark[i, 0, 1]):02d}/{int(x_mark[i, 0, 0]):04d} {int(x_mark[i, 0, 4]):02d}:00"
            hd_end_date = f"{int(x_mark[i, self.len, 2]):02d}/{int(x_mark[i, self.len, 1]):02d}/{int(x_mark[i, self.len, 0]):04d} {int(x_mark[i, self.len, 4]):02d}:00"

        else:
            hd_start_date = f"{int(x_mark[i, 0, 2]):02d}/{int(x_mark[i, 0, 1]):02d}/{int(x_mark[i, 0, 0]):04d} {int(x_mark[i, 0, 4]):02d}:{int(x_mark[i, 0, 5]):02d}"
            hd_end_date = f"{int(x_mark[i, self.len, 2]):02d}/{int(x_mark[i, self.len, 1]):02d}/{int(x_mark[i, self.len, 0]):04d} {int(x_mark[i, self.len, 4]):02d}:{int(x_mark[i, self.len, 5]):02d}"

        GT_HD_prompt = GT_HD_prompt.replace("t1", hd_start_date).replace("t2", hd_end_date)
        HD_prompt = HD_prompt.replace("t1", hd_start_date).replace("t2", hd_end_date)
        # print(GT_HD_prompt)

        GT_HD_token = self.tokenizer.encode(GT_HD_prompt, return_tensors="pt").to(self.device)
        HD_token = self.tokenizer.encode(HD_prompt, return_tensors="pt").to(self.device)
        
        gt_token_texts = self.tokenizer.convert_ids_to_tokens(GT_HD_token.squeeze(0))
        capturing = False
        gt_token_types = []
        hd_token_types = []

        for token in gt_token_texts:
            if token == 'Ġ<':
                capturing = True
                gt_token_types.append("time_series")
            elif token == '>':
                capturing = False
                gt_token_types.append("time_series")
            elif capturing:
                gt_token_types.append("time_series")
            else:
                gt_token_types.append("language")

        hd_token_texts = self.tokenizer.convert_ids_to_tokens(HD_token.squeeze(0))
        for token in hd_token_texts:
            if token == 'Ġ<':
                capturing = True
                hd_token_types.append("time_series")
            elif token == '>':
                capturing = False
                hd_token_types.append("time_series")
            elif capturing:
                hd_token_types.append("time_series")
            else:
                hd_token_types.append("language")

        return GT_HD_token, HD_token, gt_token_types, hd_token_types


    def forward(self, GT_HD_token, HD_token, gt_token_types, hd_token_types):
        GT_HD_token = GT_HD_token.to(self.device)
        HD_token = HD_token.to(self.device)

        seq_len_GT = GT_HD_token.size(0)
        seq_len_HD = HD_token.size(0)

        # causal_mask_GT = torch.tril(torch.ones((seq_len_GT, seq_len_GT), device=self.device))
        # causal_mask_HD = torch.tril(torch.ones((seq_len_HD, seq_len_HD), device=self.device))

        mask_GT = self._generate_mask(gt_token_types, seq_len_GT)
        mask_HD = self._generate_mask(hd_token_types, seq_len_HD)

        GT_HD_emb = self.gpt2(GT_HD_token, calibrated_mask=mask_GT)
        HD_emb = self.gpt2(HD_token, calibrated_mask=mask_HD)

        return GT_HD_emb, HD_emb


    def generate_embeddings(self, x, y, x_mark, y_mark):
        GT_HD = self._generate_prompt('GT_HD')
        HD = self._generate_prompt('HD')

        max_gt_token, max_hd_token = 0, 0
        GT_HD_emb_list, HD_emb_list = [], []

        for i in range(len(x)):
            for j in range(x.shape[2]):
                GT_HD_token, HD_token, gt_token_types, hd_token_types = self._prepare_prompt(GT_HD, HD, x, y, x_mark, y_mark, i, j)

                max_gt_token = max(max_gt_token, GT_HD_token.shape[1])
                max_hd_token = max(max_hd_token, HD_token.shape[1])

                GT_HD_emb_list.append((i, GT_HD_token, j))
                HD_emb_list.append((i, HD_token, j))

        # 初始化两个张量用于保存 GT 和 HD 的 prompt embeddings
        # shape: (batch_size, max_gt_token, d_model, num_experts) 或类似结构
        prompt_emb_GT = torch.zeros((len(x), max_gt_token, self.d_model, x.shape[2]), dtype=torch.float32, device=self.device)
        prompt_emb_HD = torch.zeros((len(x), max_hd_token, self.d_model, x.shape[2]), dtype=torch.float32, device=self.device)

        # 遍历 GT_HD_emb_list 和 HD_emb_list 中的数据
        for (i, GT_HD_token, j), (_, HD_token, _) in zip(GT_HD_emb_list, HD_emb_list):
            # 通过 forward 函数获取模型输出的 embedding 表示
            GT_HD_emb, HD_emb = self.forward(GT_HD_token.squeeze(0), HD_token.squeeze(0), gt_token_types, hd_token_types)

            # 增加一个维度，使形状变为 (1, seq_len, d_model)
            GT_HD_emb = GT_HD_emb.unsqueeze(0)
            HD_emb = HD_emb.unsqueeze(0)

            # 对 GT_HD_emb 进行 padding 操作，使其达到 max_gt_token 的长度
            padding_length_GT = max_gt_token - GT_HD_emb.shape[1]
            if padding_length_GT > 0:
                # 取最后一个 token 的 embedding 并重复 padding_length_GT 次
                last_token_embedding_GT = GT_HD_emb[:, -1, :].unsqueeze(1)
                padding_GT = last_token_embedding_GT.repeat(1, padding_length_GT, 1)
                # 将原始 embedding 和 padding 拼接在一起
                GT_HD_emb_padded = torch.cat([GT_HD_emb, padding_GT], dim=1)
            else:
                GT_HD_emb_padded = GT_HD_emb

            # 对 HD_emb 进行类似的 padding 操作
            padding_length_HD = max_hd_token - HD_emb.shape[1]
            if padding_length_HD > 0:
                last_token_embedding_HD = HD_emb[:, -1, :].unsqueeze(1)
                padding_HD = last_token_embedding_HD.repeat(1, padding_length_HD, 1)
                HD_emb_padded = torch.cat([HD_emb, padding_HD], dim=1)
            else:
                HD_emb_padded = HD_emb

            # 将 pad 后的 embedding 放入对应的 prompt_emb_GT 和 prompt_emb_HD 张量中
            prompt_emb_GT[i, :max_gt_token, :, j] = GT_HD_emb_padded.unsqueeze(0)
            prompt_emb_HD[i, :max_hd_token, :, j] = HD_emb_padded.unsqueeze(0)

        # 提取 GT 和 HD 的最后一层 prompt embedding
        prompt_emb_GT_1 = prompt_emb_GT[:, -1, :, :]  # shape: (batch_size, d_model, num_experts)
        prompt_emb_HD_1 = prompt_emb_HD[:, -1, :, :]  # 同上
        
        sub_out = self.sub_ac(prompt_emb_GT_1, prompt_emb_HD_1, prompt_emb_HD_1)
        sub_out = sub_out.permute(0, 2, 1).squeeze()
        return sub_out
