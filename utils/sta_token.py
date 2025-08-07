from transformers import GPT2Tokenizer

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# The input text
text = ("633532, 549773, 604004, 580329, 575664, 542536, 519414, 547187, 531790, 526723, 518121, 525260, 537190, 558382, 548573, 531250, 589930, 623077, 599366, 746485, 746230, 777397, 781234, 766753")

# Count the number of words
word_count = len(text.split())

# Tokenize the text using GPT-2 tokenizer and count the number of tokens
token_count = len(tokenizer.encode(text))

print(f"Word count: {word_count}")
print(f"Token count: {token_count}")

# Word count: 24
# Token count: 92