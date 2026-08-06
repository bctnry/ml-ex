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

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
jax.config.update("jax_default_matmul_precision", 'bfloat16')

key = jr.key(114514)

def preprocess(p):
    ii, ll, data = p
    data = jnp.float32(list(data)) / 255.0
    return ii, ll, data

class CIFAR10():
    def __init__(self):
        data_batch_1_p = pathlib.Path(os.getcwd()) / "data_batch_1.pkl"
        data_batch_2_p = pathlib.Path(os.getcwd()) / "data_batch_2.pkl"
        data_batch_3_p = pathlib.Path(os.getcwd()) / "data_batch_3.pkl"
        data_batch_4_p = pathlib.Path(os.getcwd()) / "data_batch_4.pkl"
        data_batch_5_p = pathlib.Path(os.getcwd()) / "data_batch_5.pkl"
        test_batch_p = pathlib.Path(os.getcwd()) / "test_batch.pkl"
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
        with open(str(test_batch_p), 'rb') as f:
            self.test_batch = pickle.load(f)
        self.train_dataset = jnp.concatenate([self.data_batch_1["images"], self.data_batch_2["images"], self.data_batch_3["images"], self.data_batch_4["images"], self.data_batch_5["images"]])
        self.train_label = jnp.concatenate([jnp.array(self.data_batch_1["labels"]), jnp.array(self.data_batch_2["labels"]), jnp.array(self.data_batch_3["labels"]), jnp.array(self.data_batch_4["labels"]), jnp.array(self.data_batch_5["labels"])])
        self.train_dataset_size = self.train_dataset.shape[0]
        self.t10k_dataset = self.test_batch["images"]
        self.t10k_label = jnp.array(self.test_batch["labels"])
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

augment_key = jr.key(364364)

# NOTE: the two image augment functions are written w/ AI
# (glm-5.2 via ollama). you'd see this often since this is my
# first week of doing anything related to this and have absolute
# zero numpy-fu...
def random_crop(images, key):
    pad = 4
    padded = jnp.pad(images, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='reflect')
    B = images.shape[0]
    offsets_h = jr.randint(key, (B,), 0, 2*pad+1)
    offsets_w = jr.randint(key, (B,), 0, 2*pad+1)
    C = images.shape[-1]
    def crop_one(img, h, w):
        return jax.lax.dynamic_slice(img, (h, w, 0), (32, 32, C))
    return jax.vmap(crop_one)(padded, offsets_h, offsets_w)

def random_flip(x, key):
    flip = jr.uniform(key, (x.shape[0], 1, 1, 1)) < 0.5
    return jnp.where(flip, x[:, :, ::-1, :], x)

def image_augment(x):
    global augment_key
    augment_key, k1, k2 = jr.split(augment_key, 3)
    x = random_flip(x, k1)
    x = random_crop(x, k2)
    return x
        

def get_many_train_pair(data, idx):
    # the input for nnx.Conv must be (batch_size, height, width, depth)
    # chosen = [data.get_train_pair(i) for i in idx]
    # images = jnp.stack([i[1] for i in chosen])
    # labels = jnp.array([i[0] for i in chosen])
    return image_augment(data.train_dataset[idx]), data.train_label[idx]
    # return images, labels
    # print(images.shape)
    # print(images.reshape((len(chosen), 32, 32, 3)).shape)
    # input()
    # return images.reshape((len(chosen), 32, 32, 3)), labels

def get_many_test_pair(data, idx):
    # the input for nnx.Conv must be (batch_size, height, width, depth)
    return data.t10k_dataset[idx], data.t10k_label[idx]
    # chosen = [data.get_t10k_pair(i) for i in idx]
    # images = jnp.stack([i[1] for i in chosen])
    # labels = jnp.array([i[0] for i in chosen])
    # return images.reshape((len(chosen), 32, 32, 3)), labels

def get_batch(data, batch_size, key):
    idx = jr.choice(key, cifar10.get_train_dataset_size(), (batch_size,), replace=False)
    return get_many_train_pair(data, idx)

def get_test_batch(data, batch_size, key):
    idx = jr.choice(key, cifar10.get_t10k_dataset_size(), (batch_size,), replace=False)
    return get_many_test_pair(data, idx)

class ResNet(nnx.Module):
    def __init__(self, rngs):
        self.block1_proj = nnx.Conv(3, 16, (1, 1), rngs=rngs)
        self.block1_layer1 = nnx.Conv(3, 16, (3, 3), padding='SAME', rngs=rngs)
        self.block1_bn1 = nnx.BatchNorm(16, rngs=rngs)
        self.block1_layer2 = nnx.Conv(16, 16, (3, 3), padding='SAME', rngs=rngs)
        self.block1_bn2 = nnx.BatchNorm(16, rngs=rngs)

        self.block2_proj = nnx.Conv(16, 32, (1, 1), rngs=rngs)
        self.block2_layer1 = nnx.Conv(16, 32, (3, 3), padding='SAME', rngs=rngs)
        self.block2_bn1 = nnx.BatchNorm(32, rngs=rngs)
        self.block2_layer2 = nnx.Conv(32, 32, (3, 3), padding='SAME', rngs=rngs)
        self.block2_bn2 = nnx.BatchNorm(32, rngs=rngs)

        self.block3_proj = nnx.Conv(32, 64, (1, 1), rngs=rngs)
        self.block3_layer1 = nnx.Conv(32, 64, (3, 3), padding='SAME', rngs=rngs)
        self.block3_bn1 = nnx.BatchNorm(64, rngs=rngs)
        self.block3_layer2 = nnx.Conv(64, 64, (3, 3), padding='SAME', rngs=rngs)
        self.block3_bn2 = nnx.BatchNorm(64, rngs=rngs)

        self.block4_proj = nnx.Conv(64, 128, (1, 1), rngs=rngs)
        self.block4_layer1 = nnx.Conv(64, 128, (3, 3), padding='SAME', rngs=rngs)
        self.block4_bn1 = nnx.BatchNorm(128, rngs=rngs)
        self.block4_layer2 = nnx.Conv(128, 128, (3, 3), padding='SAME', rngs=rngs)
        self.block4_bn2 = nnx.BatchNorm(128, rngs=rngs)

        self.block5_proj = nnx.Conv(128, 256, (1, 1), rngs=rngs)
        self.block5_layer1 = nnx.Conv(128, 256, (3, 3), padding='SAME', rngs=rngs)
        self.block5_bn1 = nnx.BatchNorm(256, rngs=rngs)
        self.block5_layer2 = nnx.Conv(256, 256, (3, 3), padding='SAME', rngs=rngs)
        self.block5_bn2 = nnx.BatchNorm(256, rngs=rngs)
                
        # self.layer5 = nnx.Linear(128 * 1 * 1, 10, rngs=rngs)
        self.layer5 = nnx.Linear(256 * 1 * 1, 10, rngs=rngs)

    def __call__(self, x):
        skip_x = x
        x = self.block1_layer1(x)
        x = self.block1_bn1(x)
        x = nnx.relu(x)
        x = self.block1_layer2(x)
        x = self.block1_bn2(x)
        skip_x = self.block1_proj(skip_x)
        x = x + skip_x
        x = nnx.relu(x)
        # 32 x 32 x 16

        x = nnx.max_pool(x, (2, 2), (2, 2))
        # 16 x 16 x 16

        skip_x = x
        x = self.block2_layer1(x)
        x = self.block2_bn1(x)
        x = nnx.relu(x)
        x = self.block2_layer2(x)
        x = self.block2_bn2(x)
        skip_x = self.block2_proj(skip_x)
        x = x + skip_x
        x = nnx.relu(x)
        # 16 x 16 x 32

        x = nnx.max_pool(x, (2, 2), (2, 2))
        # 8 x 8 x 32

        skip_x = x
        x = self.block3_layer1(x)
        x = self.block3_bn1(x)
        x = nnx.relu(x)
        x = self.block3_layer2(x)
        x = self.block3_bn2(x)
        skip_x = self.block3_proj(skip_x)
        x = x + skip_x
        x = nnx.relu(x)
        # 8 x 8 x 64

        x = nnx.max_pool(x, (2, 2), (2, 2))
        # 4 x 4 x 64

        skip_x = x
        x = self.block4_layer1(x)
        x = self.block4_bn1(x)
        x = nnx.relu(x)
        x = self.block4_layer2(x)
        x = self.block4_bn2(x)
        skip_x = self.block4_proj(skip_x)
        x = x + skip_x
        x = nnx.relu(x)
        # 4 x 4 x 128

        x = nnx.max_pool(x, (2,2), (2,2))
        # 2 x 2 x 128
        
        skip_x = x
        x = self.block5_layer1(x)
        x = self.block5_bn1(x)
        x = nnx.relu(x)
        x = self.block5_layer2(x)
        x = self.block5_bn2(x)
        skip_x = self.block5_proj(skip_x)
        x = x + skip_x
        x = nnx.relu(x)
        # 2 x 2 x 256

        x = nnx.avg_pool(x, (2,2), (2,2))
        # 1 x 1 x 128
        
        x = x.reshape(x.shape[0], -1)
        x = self.layer5(x)
        return x

model = ResNet(rngs=nnx.Rngs(114514))

key2 = jr.key(1919810)
n_epoch = 30
batch_size = 64
steps_per_epoch = cifar10.get_train_dataset_size() // batch_size
total_steps = n_epoch * steps_per_epoch

schedule = optax.cosine_decay_schedule(
    init_value=1e-3,
    decay_steps=total_steps,
    alpha=0.0  # final value is init_value * alpha
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=1e-4)
)
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

print("training started.")
global_step = 0
for epoch in range(n_epoch):
    key, subkey = jr.split(key2)
    perm = jr.permutation(subkey, cifar10.get_train_dataset_size())
    epoch_loss = 0

    for step in range(steps_per_epoch):
        if step % 100 == 0: print(f"Epoch {epoch+1} step {step+1} ({step / steps_per_epoch * 100:.2f}%)")
        key, subkey = jr.split(key)
        # images, labels = get_batch(mnist, batch_size, key)
        idx = perm[step * batch_size : (step + 1) * batch_size]
        images, labels = get_many_train_pair(cifar10, idx)
        state, opt_state, loss = train_step(state, opt_state, images, labels)
        epoch_loss += loss
        
        global_step += 1
    
    model = nnx.merge(graphdef, state)
    key, subkey = jr.split(key)
    test_images, test_labels = get_test_batch(cifar10, 512, key)
    model.eval()
    test_logits = model(test_images)
    model.train()
    test_acc = (test_logits.argmax(axis=1) == test_labels).mean()
    loss = epoch_loss/steps_per_epoch
    print(f"Epoch {epoch+1}: loss={loss:.4f}, testacc={test_acc:.3f}")
    if loss == 0.0:
        print('loss=0 reached. stop training...')
        break

import pathlib
import os
import orbax.checkpoint as ocp
graphdef, state = nnx.split(model)
checkpointer = ocp.PyTreeCheckpointer()
model_weight_path = pathlib.Path(os.getcwd()) / "cifar10_model.mdl"
checkpointer.save(str(model_weight_path), state)
with open('cifar10_model_graphdef.pkl', 'wb') as f:
    pickle.dump(graphdef, f)

