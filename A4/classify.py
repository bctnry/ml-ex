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
    model.eval()
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
