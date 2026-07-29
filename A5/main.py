import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import optax
import os
import pathlib
from flax import nnx

# Q: query
# K: key
# V: value
# d: row
# Q @ K^T  -->  calculates "relation strength" between Q and K
#   + would have bigger value if stronger relation
# / sqrt(d)  --> regularize
# @ V  -->  calculates "relation strength" between above and V
# softmax  -->  normalization

def masked_single_head_attention(x, Wq, Wk, Wv):
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    d = Q.shape[-1]
    a_0 = jnp.swapaxes(K, -1, -2)
    a_1 = (Q @ a_0) / jnp.sqrt(d)
    mask = jnp.tril(jnp.ones((Q.shape[-2], Q.shape[-1])))
    masked = jnp.where(mask == 0, -jnp.inf, a_1)
    a_2 = jnn.softmax(masked, axis=-1)
    return a_2 @ V

# d_head = d_model / n_heads  --> each head gets a part of the encoding
# tokens --> embeddings :: (n_token, d_model)
# splitted weights of QKV :: (n_heads, d_model, d_head)
# output of each head :: (n_token, d_head)
# combined output of each head :: (n_token, d_head * n_heads) --> (n_token, d_model)
# W_o :: (d_model, d_model)
# output result :: (n_token, d_model)
def multi_head_attention(n_heads, e, Wq, Wk, Wv, Wo):
    # requires e to be (n_token, d_model)
    # requires Wq, Wk, Wv to be of shape (n_heads, d_model, d_head)
    # requires Wo to be of shape (d_model, d_model)
    Q = jnp.einsum('td,hdk->htk', e, Wq)
    K = jnp.einsum('td,hdk->htk', e, Wk)
    V = jnp.einsum('td,hdk->htk', e, Wv)
    n_token = e.shape[0]
    d_model = Wo.shape[-1]
    d_head = d_model // n_heads
    scores = jnp.einsum('htk,hjk->htj', Q, K) / jnp.sqrt(d_head)
    mask = jnp.tril(jnp.ones((n_token, n_token)))
    scores = jnp.where(mask == 0, -jnp.inf, scores)
    weights = jnn.softmax(scores, axis=-1)
    # (n_heads, n_token, d_head)
    out = jnp.einsum('htj,hjk->htk', weights, V)
    combined = jnp.tranpsose(out, (1, 0, 2))  # (n_token, n_heads, d_head)
    combined = combined.reshape(e.shape[0], -1)  # (n_token, d_model)
    output = combined @ Wo
    return output

class FFN(nnx.Module):
    def __init__(self, d_model, rngs):
        self.fc1 = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
        self.fc2 = nnx.Linear(4 * d_model, d_model, rngs=rngs)
        
    def __call__(self, x):
        return self.fc2(nnx.gelu(self.fc1(x)))


# d_head = d_model / n_heads  --> each head gets a part of the encoding
# tokens --> embeddings :: (n_token, d_model)
# splitted weights of QKV :: (n_heads, d_model, d_head)
# output of each head :: (n_token, d_head)
# combined output of each head :: (n_token, d_head * n_heads) --> (n_token, d_model)
# W_o :: (d_model, d_model)
# output result :: (n_token, d_model)
class MultiHeadAttention(nnx.Module):
    def __init__(self, d_model, n_heads, rngs):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nnx.Linear(d_model, n_heads * self.d_head, rngs=rngs)
        self.Wk = nnx.Linear(d_model, n_heads * self.d_head, rngs=rngs)
        self.Wv = nnx.Linear(d_model, n_heads * self.d_head, rngs=rngs)
        self.Wo = nnx.Linear(n_heads * self.d_head, d_model, rngs=rngs)

    def __call__(self, x):
        batch, n_token, d_model = x.shape
        n_heads, d_head = self.n_heads, self.d_head

        # Q, K, V :: (batch, n_heads, n_token, d_head)
        Q = self.Wq(x).reshape(batch, n_token, n_heads, d_head).transpose(0, 2, 1, 3)
        K = self.Wk(x).reshape(batch, n_token, n_heads, d_head).transpose(0, 2, 1, 3)
        V = self.Wv(x).reshape(batch, n_token, n_heads, d_head).transpose(0, 2, 1, 3)

        # Q :: (batch, n_heads, n_token, d_head)
        # K :: (batch, n_heads, n_token, d_head)
        # swapped K :: (batch, n_heads, d_head, n_token)
        # scopes :: (batch, n_heads, n_token, n_token)
        scores = (Q @ jnp.swapaxes(K, -1, -2)) / jnp.sqrt(d_head)
        mask = jnp.tril(jnp.ones((n_token, n_token)))
        scores = jnp.where(mask == 0, -jnp.inf, scores)
        # weights :: (batch, n_heads, n_token, n_token)
        weights = jnn.softmax(scores, axis=-1)
        # weights :: (batch, n_heads, n_token, n_token)
        # V :: (batch, n_heads, n_token, d_head)
        out = weights @ V
        # out :: (batch, n_heads, n_token, d_head)

        # out :: (batch, n_token, n_heads, d_head)
        out = out.transpose(0, 2, 1, 3).reshape(batch, n_token, n_heads * d_head)
        # result :: (batch, n_token, d_model)
        return self.Wo(out)
        

class Transformer(nnx.Module):
    def __init__(self, d_model, n_heads, rngs):
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = MultiHeadAttention(d_model, n_heads, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn = FFN(d_model, rngs=rngs)

    def __call__(self, x):
        # skip_x = x
        # x = self.ln1(x)
        # x = self.attn(x)
        # x = skip_x + x
        # skip_x = x
        # x = self.ln2(x)
        # x = self.ffn(x)
        # x = skip_x + x
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class NanoGPT(nnx.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, rngs):
        self.W_embed = nnx.Param(jr.normal(rngs.params(), (vocab_size, d_model)) * 0.02)
        self.W_pos = nnx.Param(jr.normal(rngs.params(), (max_seq_len, d_model)) * 0.02)
        self.blocks = nnx.List([Transformer(d_model, n_heads, rngs) for _ in range(n_layers)])
        self.ln_f = nnx.LayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

    def __call__(self, token_ids):
        batch, n_token = token_ids.shape
        x = self.W_embed[token_ids]
        x = x + self.W_pos[:n_token]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # x :: (batch, n_token, d_model)
        # logits :: (batch, n_token, vocab_size)
        logits = self.head(x)
        return logits

with open('input.txt', 'r') as f:
    text = f.read()

print('training data loaded')

chars = sorted(set(text))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

data = jnp.array([char_to_id[c] for c in text])

def get_batch(data, batch_size, seq_len, key):
    max_start = len(data) - seq_len - 1
    starts = jr.randint(key, (batch_size,), 0, max_start)
    # with all the tokens [s, s+seq_len], predict [s+1, s+seq_len+1]
    # which contains the next token
    batch_input = jnp.stack([data[s:s+seq_len] for s in starts])
    batch_target = jnp.stack([data[s+1:s+seq_len+1] for s in starts])
    return batch_input, batch_target

d_model = 128
n_heads = 4
n_layers = 6
max_seq_len = 128
batch_size = 64
n_steps = 5000

model = NanoGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len,
                rngs=nnx.Rngs(114514))

schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-5,
    warmup_steps=500,
    peak_value=1e-3,
    decay_steps=n_steps,
    end_value=1e-5,
    )
optimizer = optax.adamw(learning_rate=schedule, weight_decay = 1e-2)
graphdef, state = nnx.split(model)
opt_state = optimizer.init(state)

@jax.jit
def train_step(state, opt_state, inputs, targets):
    def loss_fn(state):
        model = nnx.merge(graphdef, state)
        logits = model(inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, targets
        ).mean()
        _, new_state = nnx.split(model)
        return loss, new_state

    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state)
    updates, opt_state = optimizer.update(grads, opt_state, state)
    state = optax.apply_updates(state, updates)
    return state, opt_state, loss

key = jr.key(114514)
for step in range(n_steps):
    key, subkey = jr.split(key)
    inputs, targets = get_batch(data, batch_size, max_seq_len, subkey)
    state, opt_state, loss = train_step(state, opt_state, inputs, targets)
    if step % 25 == 0:
        print(f"Step {step}: loss={loss:.4f}")
    if loss < 1.5:
        print("loss below 1.5, stopping early")
        break

def generate_one(model, token_ids, key):
    # logits :: (batch, n_token, vocab_size)
    logits = model(token_ids)
    next_logits = logits[0, -1, :]
    next_token = jr.categorical(key, next_logits, shape=(1, ))
    return next_token
    
def generate(model, token_ids, n_new_tokens, key):
    for _ in range(n_new_tokens):
        # logits :: (batch, n_token, vocab_size)
        next_token = generate_one(model, token_ids, key)
        print(id_to_char[int(next_token[0])], end='', flush=True)
        token_ids = jnp.concatenate([token_ids, next_token[None, :]], axis=1)
        key, _ = jr.split(key)
    print('')
    return token_ids

def generate_print_only(model, token_ids, n_new_tokens, key):
    for _ in range(n_new_tokens):
        # logits :: (batch, n_token, vocab_size)
        next_token = generate_one(model, token_ids, key)
        print(id_to_char[int(next_token[0])], end='', flush=True)
        token_ids = jnp.concatenate([token_ids, next_token[None, :]], axis=1)
        key, _ = jr.split(key)
    print('')


model = nnx.merge(graphdef, state)
context = jnp.array([[char_to_id['T']]])
output_ids = generate(model, context, 500, jr.key(1919810))

