import sqlite3
import base64
import pickle
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import nnx

key = jr.key(114514)

def preprocess(p):
    ii, ll, data = p
    data = jnp.float32(list(data)) / 255.0
    return ii, ll, data

class MNIST():
    def __init__(self):
        p = "mnist.db"
        con = sqlite3.connect(p)
        cur = con.execute("SELECT COUNT(*) FROM train_dataset")
        self.train_dataset_size = cur.fetchone()[0]
        cur = con.execute("SELECT COUNT(*) FROM t10k_dataset")
        self.t10k_dataset_size = cur.fetchone()[0]
        cur = con.execute("SELECT * FROM train_dataset")
        self.train_dataset = [preprocess((i[0], i[1], base64.b64decode(i[2]))) for i in cur.fetchall()]
        cur = con.execute("SELECT * FROM t10k_dataset")
        self.t10k_dataset = [preprocess((i[0], i[1], base64.b64decode(i[2]))) for i in cur.fetchall()]
        con.close()

    def get_train_dataset_size(self):
        return self.train_dataset_size

    def get_t10k_dataset_size(self):
        return self.t10k_dataset_size

    def get_train_pair(self, i):
        return self.train_dataset[i]

    def get_t10k_pair(self, i):
        return self.t10k_dataset[i]

with open('mnist.pkl', 'rb') as f:
    mnist = pickle.load(f)
print("MNIST loaded.")


def get_many_train_pair(data, idx):
    # the input for nnx.Conv must be (batch_size, height, width, depth)
    chosen = [data.get_train_pair(i) for i in idx]
    images = jnp.stack([i[2] for i in chosen])
    labels = jnp.array([i[1] for i in chosen])
    return images.reshape((len(chosen), 28, 28, 1)), labels

def get_many_test_pair(data, idx):
    # the input for nnx.Conv must be (batch_size, height, width, depth)
    chosen = [data.get_t10k_pair(i) for i in idx]
    images = jnp.stack([i[2] for i in chosen])
    labels = jnp.array([i[1] for i in chosen])
    return images.reshape((len(chosen), 28, 28, 1)), labels

def get_batch(data, batch_size, key):
    idx = jr.choice(key, mnist.get_train_dataset_size(), (batch_size,), replace=False)
    return get_many_train_pair(data, idx)

def get_test_batch(data, batch_size, key):
    idx = jr.choice(key, mnist.get_t10k_dataset_size(), (batch_size,), replace=False)
    return get_many_test_pair(data, idx)

class CNN(nnx.Module):
    def __init__(self, rngs):
        self.layer1 = nnx.Conv(1, 64, (3, 3), padding='SAME', rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        self.layer2 = nnx.Conv(64, 128, (3, 3), padding='SAME', rngs=rngs)
        self.bn2 = nnx.BatchNorm(128, rngs=rngs)
        self.layer3 = nnx.Linear(128 * 7 * 7, 128, rngs=rngs)
        self.layer4 = nnx.Linear(128, 10, rngs=rngs)

    def __call__(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, (2,2), (2,2))
        x = self.layer2(x)
        x = self.bn2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, (2,2), (2,2))
        x = x.reshape(x.shape[0], -1)
        x = self.layer3(x)
        x = nnx.relu(x)
        x = self.layer4(x)
        return x

model = CNN(rngs=nnx.Rngs(114514))

optimizer = optax.adam(learning_rate=1e-3)
graphdef, state = nnx.split(model)
opt_state = optimizer.init(state)

@jax.jit
def train_step(state, opt_state, images, labels):
    def loss_fn(state):
        model = nnx.merge(graphdef, state)
        logits = model(images)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        ).mean()
        _, new_state = nnx.split(model)
        return loss, new_state

    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state)

    updates, opt_state = optimizer.update(grads, opt_state, state)
    state = optax.apply_updates(state, updates)

    return state, opt_state, loss

key2 = jr.key(1919810)
n_epoch = 25
batch_size = 256
steps_per_epoch = mnist.get_train_dataset_size() // batch_size

global_step = 0
for epoch in range(n_epoch):
    key, subkey = jr.split(key2)
    perm = jr.permutation(subkey, mnist.get_train_dataset_size())
    epoch_loss = 0

    for step in range(steps_per_epoch):
        print(f"Epoch {epoch+1} step {step+1} ({step / steps_per_epoch * 100:.2f}%)")
        key, subkey = jr.split(key)
        # images, labels = get_batch(mnist, batch_size, key)
        idx = perm[step * batch_size : (step + 1) * batch_size]
        images, labels = get_many_train_pair(mnist, idx)
        state, opt_state, loss = train_step(state, opt_state, images, labels)
        epoch_loss += loss
        
        global_step += 1
    
    model = nnx.merge(graphdef, state)
    key, subkey = jr.split(key)
    test_images, test_labels = get_test_batch(mnist, 32, key)
    test_logits = model(test_images)
    test_acc = (test_logits.argmax(axis=1) == test_labels).mean()
    print(f"Epoch {epoch+1}: loss={epoch_loss/steps_per_epoch:.4f},testacc={test_acc:.3f}")

import pathlib
import os
import orbax.checkpoint as ocp
graphdef, state = nnx.split(model)
checkpointer = ocp.PyTreeCheckpointer()
model_weight_path = pathlib.Path(os.getcwd()) / "mnist_model.mdl"
checkpointer.save(str(model_weight_path), state)
with open('mnist_model_graphdef.pkl', 'wb') as f:
    pickle.dump(graphdef, f)

    
