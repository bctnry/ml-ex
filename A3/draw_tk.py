### NOTE: this is done by AI using deepseek-v4-pro via ollama.  the
### training code (main.py) is handwritten but i gotta move on and i
### can't be arsed to make a demo for this.

import tkinter as tk
from tkinter import ttk
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pickle
import pathlib
import os
from flax import nnx
from PIL import Image, ImageDraw


# --- Load model ---

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

with open('mnist_model_graphdef.pkl', 'rb') as f:
    graphdef = pickle.load(f)
checkpointer = ocp.PyTreeCheckpointer()
model_weight_path = pathlib.Path(os.getcwd()) / "mnist_model.mdl"
state = checkpointer.restore(str(model_weight_path))
model = nnx.merge(graphdef, state)

# --- App ---
GRID = 28
SCALE = 12  # each pixel displayed as 12x12
DISPLAY_SIZE = GRID * SCALE  # 336

root = tk.Tk()
root.title("MNIST Digit Recognizer")
root.configure(bg='#1e1e1e')

# Left frame: drawing canvas
left = ttk.Frame(root)
left.pack(side=tk.LEFT, padx=10, pady=10)

# The actual data is a 28x28 image
canvas_img = Image.new('L', (GRID, GRID), 255)
draw_img = ImageDraw.Draw(canvas_img)

# Display canvas is scaled up
tk_canvas = tk.Canvas(left, width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                       bg='white', cursor='cross', highlightthickness=1,
                       highlightbackground='#555')
tk_canvas.pack()

btn_frame = ttk.Frame(left)
btn_frame.pack(pady=(8, 0))
ttk.Button(btn_frame, text="Predict", command=lambda: predict()).pack(side=tk.LEFT, padx=4)
ttk.Button(btn_frame, text="Clear", command=lambda: clear()).pack(side=tk.LEFT, padx=4)

# Right frame: prediction bars
right = ttk.Frame(root)
right.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

pred_label = tk.Label(right, text="?", font=('Helvetica', 48, 'bold'),
                       fg='white', bg='#1e1e1e')
pred_label.pack(pady=(0, 10))

bar_frame = ttk.Frame(right)
bar_frame.pack(fill=tk.BOTH, expand=True)

BAR_WIDTH = 30
BAR_MAX_H = 200
bars = []
bar_texts = []
for i in range(10):
    col_frame = ttk.Frame(bar_frame)
    col_frame.pack(side=tk.LEFT, padx=2)

    # Bar
    bar = tk.Canvas(col_frame, width=BAR_WIDTH, height=BAR_MAX_H,
                     bg='#1e1e1e', highlightthickness=0)
    bar.pack()
    bars.append(bar)

    # Digit label
    tk.Label(col_frame, text=str(i), fg='#8b949e', bg='#1e1e1e',
             font=('Helvetica', 10)).pack()

    # Probability text
    txt = tk.Label(col_frame, text='', fg='#8b949e', bg='#1e1e1e',
                    font=('Helvetica', 8))
    txt.pack()
    bar_texts.append(txt)

# --- Drawing ---
BRUSH = 2

def paint(event):
    gx = event.x // SCALE
    gy = event.y // SCALE
    for dx in range(BRUSH):
        for dy in range(BRUSH):
            px, py = gx + dx, gy + dy
            if 0 <= px < GRID and 0 <= py < GRID:
                x0, y0 = px * SCALE, py * SCALE
                x1, y1 = x0 + SCALE, y0 + SCALE
                tk_canvas.create_rectangle(x0, y0, x1, y1, fill='black', outline='black')
                draw_img.point((px, py), fill=0)

tk_canvas.bind('<B1-Motion>', paint)
tk_canvas.bind('<Button-1>', paint)

# --- Prediction ---
def predict():
    arr = np.array(canvas_img, dtype=np.float32) / 255.0
    x = jnp.array(arr.reshape(1, 28, 28, 1))  # (batch, H, W, C)
    logits = model(x)
    probs = jax.nn.softmax(logits[0])

    best = int(jnp.argmax(probs))
    pred_label.config(text=str(best))

    for i in range(10):
        p = float(probs[i])
        h = int(p * BAR_MAX_H)
        bars[i].delete('all')
        color = '#3fb950' if i == best else '#58a6ff'
        bars[i].create_rectangle(0, BAR_MAX_H - h, BAR_WIDTH, BAR_MAX_H,
                                  fill=color, outline='')
        bar_texts[i].config(text=f'{p:.2f}')

# --- Clear ---
def clear():
    global canvas_img, draw_img
    canvas_img = Image.new('L', (GRID, GRID), 255)
    draw_img = ImageDraw.Draw(canvas_img)
    tk_canvas.delete('all')
    pred_label.config(text='?')
    for i in range(10):
        bars[i].delete('all')
        bar_texts[i].config(text='')

root.mainloop()































































































































































































































































































































