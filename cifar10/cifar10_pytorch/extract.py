import os
import pickle
import numpy

with open('batches.meta.txt', 'r') as f:
    labels_s = f.read()
labels = labels_s.strip().split('\n')

images = []
labels = []
print('extracting data_batch_1.bin...')
counter = 0
with open('data_batch_1.bin', 'rb') as f:
    image = []
    while True:
        if counter % 100 == 0: print(f'extraction progress ({counter}/10000) ({counter/10000*100:.4f}%)')
        label = f.read(1)
        if label == b'': break
        data_red = f.read(1024)
        data_green = f.read(1024)
        data_blue = f.read(1024)
        for i in range(32):
            row = []
            for j in range(32):
                row.append([data_red[i]/255.0, data_green[i]/255.0, data_blue[i]/255.0])
            image.append(row)
        image_array = numpy.array(image)
        images.append(image_array)
        labels.append(label[0])
        image = []
        counter += 1
images = numpy.array(images)
labels = numpy.array(labels)

with open('data_batch_1.pkl', 'wb') as f:
    pickle.dump({"images": images, "labels": labels}, f)

print("batch 1 complete.")

images = []
labels = []
print('extracting data_batch_2.bin...')
counter = 0
with open('data_batch_2.bin', 'rb') as f:
    image = []
    while True:
        if counter % 100 == 0: print(f'extraction progress ({counter}/10000) ({counter/10000*100:.4f}%)')
        label = f.read(1)
        if label == b'': break
        data_red = f.read(1024)
        data_green = f.read(1024)
        data_blue = f.read(1024)
        for i in range(32):
            row = []
            for j in range(32):
                row.append([data_red[i]/255.0, data_green[i]/255.0, data_blue[i]/255.0])
            image.append(row)
        image_array = numpy.array(image)
        images.append(image_array)
        labels.append(label[0])
        image = []
        counter += 1
images = numpy.array(images)
labels = numpy.array(labels)

with open('data_batch_2.pkl', 'wb') as f:
    pickle.dump({"images": images, "labels": labels}, f)

print("batch 2 complete.")

images = []
labels = []
print('extracting data_batch_3.bin...')
counter = 0
with open('data_batch_3.bin', 'rb') as f:
    image = []
    while True:
        if counter % 100 == 0: print(f'extraction progress ({counter}/10000) ({counter/10000*100:.4f}%)')
        label = f.read(1)
        if label == b'': break
        data_red = f.read(1024)
        data_green = f.read(1024)
        data_blue = f.read(1024)
        for i in range(32):
            row = []
            for j in range(32):
                row.append([data_red[i]/255.0, data_green[i]/255.0, data_blue[i]/255.0])
            image.append(row)
        image_array = numpy.array(image)
        images.append(image_array)
        labels.append(label[0])
        image = []
        counter += 1
images = numpy.array(images)
labels = numpy.array(labels)

with open('data_batch_3.pkl', 'wb') as f:
    pickle.dump({"images": images, "labels": labels}, f)

print("batch 3 complete.")

images = []
labels = []
print('extracting data_batch_4.bin...')
counter = 0
with open('data_batch_4.bin', 'rb') as f:
    image = []
    while True:
        if counter % 100 == 0: print(f'extraction progress ({counter}/10000) ({counter/10000*100:.4f}%)')
        label = f.read(1)
        if label == b'': break
        data_red = f.read(1024)
        data_green = f.read(1024)
        data_blue = f.read(1024)
        for i in range(32):
            row = []
            for j in range(32):
                row.append([data_red[i]/255.0, data_green[i]/255.0, data_blue[i]/255.0])
            image.append(row)
        image_array = numpy.array(image)
        images.append(image_array)
        labels.append(label[0])
        image = []
        counter += 1
images = numpy.array(images)
labels = numpy.array(labels)

with open('data_batch_4.pkl', 'wb') as f:
    pickle.dump({"images": images, "labels": labels}, f)

print("batch 4 complete.")

images = []
labels = []
print('extracting data_batch_5.bin...')
counter = 0
with open('data_batch_5.bin', 'rb') as f:
    image = []
    while True:
        if counter % 100 == 0: print(f'extraction progress ({counter}/10000) ({counter/10000*100:.4f}%)')
        label = f.read(1)
        if label == b'': break
        data_red = f.read(1024)
        data_green = f.read(1024)
        data_blue = f.read(1024)
        for i in range(32):
            row = []
            for j in range(32):
                row.append([data_red[i]/255.0, data_green[i]/255.0, data_blue[i]/255.0])
            image.append(row)
        image_array = numpy.array(image)
        images.append(image_array)
        labels.append(label[0])
        image = []
        counter += 1
images = numpy.array(images)
labels = numpy.array(labels)

with open('data_batch_5.pkl', 'wb') as f:
    pickle.dump({"images": images, "labels": labels}, f)

print("batch 5 complete.")

images = []
labels = []
print('extracting test_batch.bin...')
counter = 0
with open('test_batch.bin', 'rb') as f:
    image = []
    while True:
        if counter % 100 == 0: print(f'extraction progress ({counter}/10000) ({counter/10000*100:.4f}%)')
        label = f.read(1)
        if label == b'': break
        data_red = f.read(1024)
        data_green = f.read(1024)
        data_blue = f.read(1024)
        for i in range(32):
            row = []
            for j in range(32):
                row.append([data_red[i]/255.0, data_green[i]/255.0, data_blue[i]/255.0])
            image.append(row)
        image_array = numpy.array(image)
        images.append(image_array)
        labels.append(label[0])
        image = []
        counter += 1
images = numpy.array(images)
labels = numpy.array(labels)

with open('test_batch.pkl', 'wb') as f:
    pickle.dump({"images": images, "labels": labels}, f)

print("test batch complete.")


