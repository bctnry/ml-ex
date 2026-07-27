import sqlite3
import base64
import pickle
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import os
import pathlib
from flax import nnx

key = jr.key(114514)

def preprocess(p):
    ii, ll, data = p
    data = jnp.float32(list(data)) / 255.0
    return ii, ll, data

class CIFAR10():
    def __init__(self):
        p = "cifar-10-batches-bin"
        data_batch_1_p = pathlib.Path(os.getcwd()) / p / "data_batch_1.pkl"
        data_batch_2_p = pathlib.Path(os.getcwd()) / p / "data_batch_2.pkl"
        data_batch_3_p = pathlib.Path(os.getcwd()) / p / "data_batch_3.pkl"
        data_batch_4_p = pathlib.Path(os.getcwd()) / p / "data_batch_4.pkl"
        data_batch_5_p = pathlib.Path(os.getcwd()) / p / "data_batch_5.pkl"
        with open(str(data_batch_1_p), 'rb') as f:
            self.data_batch_1 = pickle.load(f)
        with open(str(data_batch_2_p), 'rb') as f:
            self.data_batch_2 = pickle.load(f)
        with open(str(data_batch_3_p), 'rb') as f:
            self.data_batch_3 = pickle.load(f)
        with open(str(data_batch_4_p), 'rb') as f:
            self.data_batch_4 = pickle.load(f)
        with open(str(data_batch_5_p), 'rb') as f:
            self.data_batch_5 = pickle.load(f)
        self.train_dataset = jnp.concatenate([self.data_batch_1["images"], self.data_batch_2["images"], self.data_batch_3["images"], self.data_batch_4["images"]])
        self.train_label = jnp.concatenate([jnp.array(self.data_batch_1["labels"]), jnp.array(self.data_batch_2["labels"]), jnp.array(self.data_batch_3["labels"]), jnp.array(self.data_batch_4["labels"])])
        self.train_dataset_size = self.train_dataset.shape[0]
        self.t10k_dataset = self.data_batch_5["images"]
        self.t10k_label = jnp.array(self.data_batch_5["labels"])
        self.t10k_dataset_size = self.t10k_dataset.shape[0]

    def get_train_dataset_size(self):
        return self.train_dataset_size

    def get_t10k_dataset_size(self):
        return self.t10k_dataset_size

    def get_train_pair(self, i):
        return (self.train_label[i], self.train_dataset[i])

    def get_t10k_pair(self, i):
        return (self.t10k_label[i], self.t10k_dataset[i])

cifar10 = CIFAR10()
print("CIFAR-10 loaded.")


def get_many_train_pair(data, idx):
    # the input for nnx.Conv must be (batch_size, height, width, depth)
    chosen = [data.get_train_pair(i) for i in idx]
    images = jnp.stack([i[1] for i in chosen])
    labels = jnp.array([i[0] for i in chosen])
    return images, labels
    # print(images.shape)
    # print(images.reshape((len(chosen), 32, 32, 3)).shape)
    # input()
    # return images.reshape((len(chosen), 32, 32, 3)), labels

def get_many_test_pair(data, idx):
    # the input for nnx.Conv must be (batch_size, height, width, depth)
    chosen = [data.get_t10k_pair(i) for i in idx]
    images = jnp.stack([i[1] for i in chosen])
    labels = jnp.array([i[0] for i in chosen])
    return images.reshape((len(chosen), 32, 32, 3)), labels

def get_batch(data, batch_size, key):
    idx = jr.choice(key, cifar10.get_train_dataset_size(), (batch_size,), replace=False)
    return get_many_train_pair(data, idx)

def get_test_batch(data, batch_size, key):
    idx = jr.choice(key, cifar10.get_t10k_dataset_size(), (batch_size,), replace=False)
    return get_many_test_pair(data, idx)

class CNN(nnx.Module):
    def __init__(self, rngs):
        self.layer1 = nnx.Conv(3, 64, (3, 3), padding='SAME', rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        self.layer2 = nnx.Conv(64, 128, (3, 3), padding='SAME', rngs=rngs)
        self.bn2 = nnx.BatchNorm(128, rngs=rngs)
        self.layer3 = nnx.Linear(128 * 8 * 8, 128, rngs=rngs)
        self.layer4 = nnx.Linear(128, 64, rngs=rngs)
        self.layer5 = nnx.Linear(64, 10, rngs=rngs)

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
        x = nnx.relu(x)
        x = self.layer5(x)
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
n_epoch = 40
batch_size = 256
steps_per_epoch = cifar10.get_train_dataset_size() // batch_size

print("training started.")
global_step = 0
for epoch in range(n_epoch):
    key, subkey = jr.split(key2)
    perm = jr.permutation(subkey, cifar10.get_train_dataset_size())
    epoch_loss = 0

    for step in range(steps_per_epoch):
        print(f"Epoch {epoch+1} step {step+1} ({step / steps_per_epoch * 100:.2f}%)")
        key, subkey = jr.split(key)
        # images, labels = get_batch(mnist, batch_size, key)
        idx = perm[step * batch_size : (step + 1) * batch_size]
        images, labels = get_many_train_pair(cifar10, idx)
        state, opt_state, loss = train_step(state, opt_state, images, labels)
        epoch_loss += loss
        
        global_step += 1
    
    model = nnx.merge(graphdef, state)
    key, subkey = jr.split(key)
    test_images, test_labels = get_test_batch(cifar10, 32, key)
    test_logits = model(test_images)
    test_acc = (test_logits.argmax(axis=1) == test_labels).mean()
    print(f"Epoch {epoch+1}: loss={epoch_loss/steps_per_epoch:.4f},testacc={test_acc:.3f}")

import pathlib
import os
import orbax.checkpoint as ocp
graphdef, state = nnx.split(model)
checkpointer = ocp.PyTreeCheckpointer()
model_weight_path = pathlib.Path(os.getcwd()) / "cifar10_model.mdl"
checkpointer.save(str(model_weight_path), state)
with open('cifar10_model_graphdef.pkl', 'wb') as f:
    pickle.dump(graphdef, f)

    
