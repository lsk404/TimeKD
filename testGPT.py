from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2", attn_implementation="eager",
                                              output_attentions=True, output_hidden_states=True)  #attn_implementation="sdpa" OR "eager"
        