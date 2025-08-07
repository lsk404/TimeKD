from transformers import GPT2Tokenizer, GPT2Model

# 指定保存的目录（当前路径下的文件夹名，例如 "gpt2_model"）
save_model_directory = "./gpt2_model"
save_tokenizer_directory = "./gpt2_tokenizer"

# 下载并保存 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained(save_tokenizer_directory)

# 下载并保存模型
gpt2 = GPT2Model.from_pretrained(
    "gpt2", 
    attn_implementation="eager",
    output_attentions=True, 
    output_hidden_states=True
)
gpt2.save_pretrained(save_model_directory)

print(f"保存完成")