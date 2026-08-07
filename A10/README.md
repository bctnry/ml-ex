dataset used:
+ `karpathy/tinystories-gpt4-clean`: https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean
+ `karpathy/tiny_shakespeare`: https://huggingface.co/datasets/karpathy/tiny_shakespeare

prepare like this:
1. rename tinyshakespeare to `input.txt` and run `python ./prepare_data_plaintext.py`. got `tokens.bin`, rename to `tinyshakespeare_tokens.bin`.
2. download the parquet file of tinystores into its dedicated directory. run `prepare_data.py` (which expects the parquet file to be under `tinystories/`; you can change it if you want). got `tokens.bin`, rename to `tinystories_tokens.bin`.
3. run `torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29500 ./main.py`. actual arg configs are up to you. currently code calls for 20k steps, which might be too much and probably caused overfitting.
4. (optional) you can edit and run `python3 ./generate.py` and try the base model.
5. run `torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29500 ./lora.py`. actual arg configs are up to you.
6. run `python3 ./generate_lora.py` to see the difference.

requires:
+ `numpy` and `torch` (of course)
+ `tiktoken` (for tokenizer)
+ `pyarrow` (for reading parquet file)

