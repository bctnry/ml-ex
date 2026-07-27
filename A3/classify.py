### Note: this is written by AI (deepseek-v4-pro via ollama)

import sys
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pickle
import pathlib
import os
from flax import nnx
from PIL import Image

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

# --- Load model ---
with open('cifar10_model_graphdef.pkl', 'rb') as f:
    graphdef = pickle.load(f)
checkpointer = ocp.PyTreeCheckpointer()
model_weight_path = pathlib.Path(os.getcwd()) / "cifar10_model.mdl"
state = checkpointer.restore(str(model_weight_path))
model = nnx.merge(graphdef, state)

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def classify(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    x = jnp.array(arr.reshape(1, 32, 32, 3))
    logits = model(x)
    probs = jax.nn.softmax(logits[0])

    ranked = sorted(zip(CIFAR10_CLASSES, probs), key=lambda p: p[1], reverse=True)
    for name, p in ranked:
        bar = '█' * int(p * 40)
        print(f'  {name:<12s} {p:.3f}  {bar}')
    print(f'\n  → {ranked[0][0]} ({ranked[0][1]:.1%})')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python classify.py <image_path>')
        sys.exit(1)
    classify(sys.argv[1])
