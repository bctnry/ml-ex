import math
import datetime
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

key = jr.key(114514)
def random_key():
    global key
    key, subkey = jr.split(key)
    return subkey

group_a = jr.normal(random_key(), (100,1)) + 7
group_b = jr.normal(random_key(), (100,1)) + 1

# sigmoid(z)
# z(x) = w * x + b
def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))
def subj(w, b, x):
    return sigmoid(x @ w + b)
def loss(w, b, x, y):
    pred = subj(w, b, x)
    # cross-entropy...
    # the 1e-8 is probably here to prevent log from getting zero/negative input.
    # l_k = - y_k ln(p_k) - (1 - y_k) ln (1 - p_k)
    # where p_k is the `pred` above and `y_k` is y.
    # thanks to how JAX works we can just do all of these things.
    # loss = jnp.mean(- y * jnp.log(pred + 1e-8) - (1 - y) * jnp.log(1 - pred + 1e-8))
    loss = jnp.mean(- y * jnp.log(pred) - (1 - y) * jnp.log(1 - pred))
    return loss

# grad against w and b.
# grad_loss = jax.grad(loss, argnums=(0, 1))
grad_loss = jax.jit(jax.grad(loss, argnums=(0, 1)))

X = jnp.concatenate([group_a, group_b])
y = jnp.array([0.0] * 100 + [1.0] * 100)

w = jnp.zeros((1,))
b = 0.0
learning_rate = 0.01
for step in range(5000):
    gw, gb = grad_loss(w, b, X, y)
    w = w - learning_rate * gw
    b = b - learning_rate * gb
    if step % 100 == 0:
        print(f"step {step}, loss {loss(w, b, X, y):.4f}")

fig, ax = plt.subplots()
# ax.scatter(data=group_a)
ax.scatter(group_a,
           [0] * 100,
           c="blue")
ax.scatter(group_b,
           [1] * 100,
           c="red")
ax.grid(True)
fig.tight_layout()
x_vals = jnp.linspace(-4, 8, 100)
y_vals = sigmoid(w[0] * x_vals + b)
ax.plot(x_vals, y_vals, 'r--', label='regression result')
plt.show()


