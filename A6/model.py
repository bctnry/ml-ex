import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import optax
from flax import nnx

class FFN(nnx.Module):
    def __init__(self, d_model, rngs):
        self.fc1 = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
        self.fc2 = nnx.Linear(4 * d_model, d_model, rngs=rngs)
        
    def __call__(self, x):
        return self.fc2(nnx.gelu(self.fc1(x)))

# rotary position embedding.
# + the idea of position embedding is to add in information about positions,
#   because self-attention is position-agnostic, meaning where a certain
#   appears in the text does not make a difference, which is wrong for
#   (natural) languages.
# + absolute position embedding is able to manage a certain token appear
#   at a certain absolute position but isn't able to express relative info,
#   which is one big flaw because the same token could appear at any position
#   inside texts of any length.
# + for rotary position embedding, we rotate the embedding vector.
# + for any two vectors, if we rotate one by a * theta and another by b * theta
#   (where a and b are some representation of absolute position indicies), the
#   (angular) distance between them is solely dependent on cos((a - b) * theta),
#   which is relative because we're asking about a - b. this allows simple
#   inclusion of relative position info.
# + technically "rotating by an angle" only makes sense when it's rotating with
#   respect to a 2D plane, but Q and K is of length d_head for each vector,
#   i.e. "d_head"-D. the idea is to split this "d_head"-D into d_head/2 2Ds
#   and rotate them.
# + we rotate them at different angles to capture "multiple sense" of relative info.
def rope(x, pos):
    # x :: (batch, n_heads, n_token, d_head)
    d_head = x.shape[-1]
    # freqs :: (d_head/2,)  (one per pair)
    freqs = 1.0 / (10000.0 ** (jnp.arange(0, d_head, 2) / d_head))
    # angles :: (seq, d_head/2)
    angles = pos[:, None] * freqs[None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # 2D rotation:
    # [x']  =  [cos(theta) -sin(theta)] [x]
    # [y']  =  [sin(theta)  cos(theta)] [y]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = jnp.empty_like(x)
    out = out.at[..., 0::2].set(out_even)
    out = out.at[..., 1::2].set(out_odd)
    return out
    
    
# d_head = d_model / n_heads  --> each head gets a part of the encoding
# tokens --> embeddings :: (n_token, d_model)
# splitted weights of QKV :: (n_heads, d_model, d_head)
# output of each head :: (n_token, d_head)
# combined output of each head :: (n_token, d_head * n_heads) --> (n_token, d_model)
# W_o :: (d_model, d_model)
# output result :: (n_token, d_model)
class MultiHeadAttention(nnx.Module):
    def __init__(self, max_seq_len, d_model, n_heads, rngs):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.Wq = nnx.Linear(d_model, n_heads * self.d_head, rngs=rngs)
        self.Wk = nnx.Linear(d_model, n_heads * self.d_head, rngs=rngs)
        self.Wv = nnx.Linear(d_model, n_heads * self.d_head, rngs=rngs)
        self.Wo = nnx.Linear(n_heads * self.d_head, d_model, rngs=rngs)

    def __call__(self, x, k_cache=None, v_cache=None, infer=False, pos_offset=0):
        batch, n_token, d_model = x.shape
        n_heads, d_head = self.n_heads, self.d_head

        # Q, K, V :: (batch, n_heads, n_token, d_head)
        Q = self.Wq(x).reshape(batch, n_token, n_heads, d_head).transpose(0, 2, 1, 3)
        K = self.Wk(x).reshape(batch, n_token, n_heads, d_head).transpose(0, 2, 1, 3)
        V = self.Wv(x).reshape(batch, n_token, n_heads, d_head).transpose(0, 2, 1, 3)

        positions = jnp.arange(n_token) + pos_offset
        Q = rope(Q, positions)
        K = rope(K, positions)

        if k_cache is not None:
            K = jnp.concatenate([k_cache, K], axis=2)
            V = jnp.concatenate([v_cache, V], axis=2)

        # Q :: (batch, n_heads, n_token, d_head)
        # K :: (batch, n_heads, n_token, d_head)
        # swapped K :: (batch, n_heads, d_head, n_token)
        # scopes :: (batch, n_heads, n_token, n_token)
        scores = (Q @ jnp.swapaxes(K, -1, -2)) / jnp.sqrt(d_head)
        if not infer:
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
        return self.Wo(out), K, V
        

class Transformer(nnx.Module):
    def __init__(self, max_seq_len, d_model, n_heads, rngs):
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = MultiHeadAttention(max_seq_len, d_model, n_heads, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn = FFN(d_model, rngs=rngs)

    def __call__(self, x, k_cache=None, v_cache=None, infer=False, pos_offset=0):
        attn, k_new, v_new = self.attn(self.ln1(x), k_cache=k_cache, v_cache=v_cache, infer=infer, pos_offset=pos_offset)
        x = x + attn
        x = x + self.ffn(self.ln2(x))
        return x, k_new, v_new

    
class NanoGPT(nnx.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, rngs):
        self.W_embed = nnx.Param(jr.normal(rngs.params(), (vocab_size, d_model)) * 0.02)
        self.blocks = nnx.List([Transformer(max_seq_len, d_model, n_heads, rngs) for _ in range(n_layers)])
        self.ln_f = nnx.LayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

    def __call__(self, token_ids, k_cache=None, v_cache=None, infer=False, pos_offset=0):
        batch, n_token = token_ids.shape
        x = self.W_embed.value[token_ids]
        k_caches = []
        v_caches = []
        for i, block in enumerate(self.blocks):
            x, k, v = block(
                x,
                k_cache=k_cache[i] if k_cache is not None else None,
                v_cache=v_cache[i] if v_cache is not None else None,
                infer=infer,
                pos_offset=pos_offset
            )
            k_caches.append(k)
            v_caches.append(v)
        x = self.ln_f(x)
        # x :: (batch, n_token, d_model)
        # logits :: (batch, n_token, vocab_size)
        logits = self.head(x)
        return logits, k_caches, v_caches

