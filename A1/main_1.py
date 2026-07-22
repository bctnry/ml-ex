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

group_a = jr.normal(random_key(), (100,2)) + jnp.array([-2, 2])
group_b = jr.normal(random_key(), (100,2)) + jnp.array([2, -2])

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

w = jnp.zeros((2,))
b = 0.0
learning_rate = 0.005
for step in range(500):
    if step == 5:
        w1 = w
        b1 = b
    elif step == 10:
        w2 = w
        b2 = b
    gw, gb = grad_loss(w, b, X, y)
    w = w - learning_rate * gw
    b = b - learning_rate * gb
    if step % 100 == 0:
        print(f"step {step}, loss {loss(w, b, X, y):.4f}")

fig, ax = plt.subplots()
# ax.scatter(data=group_a)
ax.scatter(group_a[:, 0],
           group_a[:, 1],
           c="blue")
ax.scatter(group_b[:, 0],
           group_b[:, 1],
           c="red")
ax.grid(True)
fig.tight_layout()
x_vals = jnp.linspace(-6, 6, 100)
y_vals = -(w[0] * x_vals + b) / w[1]
y2_vals = -(w1[0] * x_vals + b) / w1[1]
y3_vals = -(w2[0] * x_vals + b) / w2[1]
ax.plot(x_vals, y_vals, 'r--', label='regression result')
ax.plot(x_vals, y2_vals, 'g--', label='regression result')
ax.plot(x_vals, y3_vals, 'b--', label='regression result')
plt.show()



















