import torch
import tiktoken
import pyarrow.dataset as pds

# converts .parquet datasets into something DatasetManager needs.

print('creating tokenizer...')
Tokenizer = tiktoken.get_encoding('gpt2')
print('opening dataset...')
with open('input.txt', 'r') as f:
    ds = f.read()
result = open('tokens.bin', 'wb')
token_list = Tokenizer.encode(ds)
token_list.append(Tokenizer.eot_token)
for k in token_list:
    byte1 = k%256
    byte2 = (k//256)%256
    result.write(bytes([byte1, byte2]))
result.close()

