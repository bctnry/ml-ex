### NOTE: requires the OG dataset files gunzipped.  this produces a
### sqlite3 db `mnist.db` (~91MB) and a pickle file `mnist.pkl`
### (~212MB). the latter requires the class definition and may load a
### bit faster than the sqlite3.

import sqlite3
import base64

con = sqlite3.connect('mnist.db')
cur = con.cursor()
cur.execute("DROP TABLE IF EXISTS train_dataset")
cur.execute("DROP TABLE IF EXISTS t10k_dataset")
con.commit()

cur.execute("CREATE TABLE IF NOT EXISTS train_dataset(id, label, data)")

label_file_path = 'train-labels-idx1-ubyte'
image_file_path = 'train-images-idx3-ubyte'
index_file_path = 'train_image/index'

label_file = open(label_file_path, 'rb')
image_file = open(image_file_path, 'rb')

if label_file.read(4) != b'\x00\x00\x08\x01':
    raise ValueError('wrong magic number for label file')

if image_file.read(4) != b'\x00\x00\x08\x03':
    raise ValueError('wrong magic number for image file')

def qw2n_be(s):
    return s[0] << 24 | s[1] << 16 | s[2] << 8 | s[3]

n_label = qw2n_be(label_file.read(4))
n_image = qw2n_be(image_file.read(4))

if n_label != n_image:
    raise ValueError(f'n of labels ({n_label}) and n of images ({n_image}) mismatch')

n_row = qw2n_be(image_file.read(4))
n_col = qw2n_be(image_file.read(4))
for i in range(n_label):
    label = label_file.read(1)[0]
    data = []
    for _ in range(n_row):
        for _ in range(n_col):
            data.append(255 - image_file.read(1)[0])
    cur.execute("INSERT INTO train_dataset(id, label, data) VALUES (:id, :label, :data)", { "id": i, "label": label, "data": base64.b64encode(bytes(data)).decode('utf-8') })

label_file.close()
image_file.close()
con.commit()

cur.execute("CREATE TABLE IF NOT EXISTS t10k_dataset(id, label, data)")

label_file_path = 't10k-labels-idx1-ubyte'
image_file_path = 't10k-images-idx3-ubyte'

label_file = open(label_file_path, 'rb')
image_file = open(image_file_path, 'rb')

if label_file.read(4) != b'\x00\x00\x08\x01':
    raise ValueError('wrong magic number for label file')

if image_file.read(4) != b'\x00\x00\x08\x03':
    raise ValueError('wrong magic number for image file')

n_label = qw2n_be(label_file.read(4))
n_image = qw2n_be(image_file.read(4))

if n_label != n_image:
    raise ValueError(f'n of labels ({n_label}) and n of images ({n_image}) mismatch')

n_row = qw2n_be(image_file.read(4))
n_col = qw2n_be(image_file.read(4))
for i in range(n_label):
    label = label_file.read(1)[0]
    data = []
    for _ in range(n_row):
        for _ in range(n_col):
            data.append(255 - image_file.read(1)[0])
    cur.execute("INSERT INTO t10k_dataset(id, label, data) VALUES (:id, :label, :data)", { "id": i, "label": label, "data": base64.b64encode(bytes(data)).decode('utf-8') })

label_file.close()
image_file.close()
con.commit()
con.close()

## pickled object for fast loading.

import jax.numpy as jnp
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

mnist = MNIST()
import pickle
with open('mnist.pkl', 'wb') as f:
    pickle.dump(mnist, f)
        
