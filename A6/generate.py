import pathlib
import os
import jax
import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as ocp
from flax import nnx
from model import NanoGPT

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
jax.config.update("jax_default_matmul_precision", 'bfloat16')

d_model = 128
n_heads = 4
n_layers = 6
max_seq_len = 256

with open('input.txt', 'r') as f:
    text = f.read()
print('training data loaded')
chars = sorted(set(text))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

model = NanoGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len,
                rngs=nnx.Rngs(114514))
graphdef, state = nnx.split(model)
checkpointer = ocp.PyTreeCheckpointer()
model_weight_path = pathlib.Path(os.getcwd()) / "nanogpt_model.mdl"
state = checkpointer.restore(str(model_weight_path))
model = nnx.merge(graphdef, state)

def gen_step(state, token_ids, k_caches, v_caches, key, temperature, pos_offset, top_k):
    model = nnx.merge(graphdef, state)
    logits, k_caches, v_caches = model(token_ids, infer=True,
                                       k_cache=k_caches, v_cache=v_caches,
                                       pos_offset=pos_offset)
    next_logits = logits[0, -1, :] / temperature
    topk_vals = jnp.sort(next_logits.reshape(vocab_size))[-top_k:]
    threshold = topk_vals[0]
    next_logits = jnp.where(next_logits < threshold, -jnp.inf, next_logits)
    next_token = jr.categorical(key, next_logits, shape=(1,))
    return next_token, k_caches, v_caches
gen_step = jax.jit(gen_step, static_argnames=['top_k'])

def generate(model, token_ids, n_new_tokens, key, temperature=1.0, top_k=25):
    # k_caches, v_caches = (n_blocks, batch, n_heads, n_token, d_head)
    graphdef, state = nnx.split(model)
    next_token, k_caches, v_caches = gen_step(state, token_ids, None, None, key,
                                              temperature, 0, top_k)
    print(id_to_char[int(next_token[0])], end='', flush=True)
    pos = len(token_ids)
    token_ids = jnp.concatenate([token_ids, next_token[None, :]], axis=1)
    key, _ = jr.split(key)
    for _ in range(n_new_tokens):
        # logits :: (batch, n_token, vocab_size)
        next_token, k_caches, v_caches = gen_step(state, next_token[None, :],
                                                  k_caches, v_caches, key,
                                                  temperature, pos, top_k)
        print(id_to_char[int(next_token[0])], end='', flush=True)
        token_ids = jnp.concatenate([token_ids, next_token[None, :]], axis=1)
        if k_caches[0].shape[2] >= max_seq_len:
            k_caches = [k[:, :, -max_seq_len:, :] for k in k_caches]
            v_caches = [v[:, :, -max_seq_len:, :] for v in v_caches]
        key, _ = jr.split(key)
        pos += 1
    print('')
    return token_ids

model = nnx.merge(graphdef, state)
context = jnp.array([[char_to_id[i] for i in '''JULIET''']])
output_ids = generate(model, context, 500, jr.key(1919810))


