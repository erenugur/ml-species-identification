import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "/content/drive/My Drive/AI Club Species Identification Project/Datasets/Training Data (2000 files)"
CATEGORIES = ["Muscicapa striata (1000 spectrograms)", "Tringa ochropus (1000 spectrograms)"]

for category in CATEGORIES:
  path = os.path.join(DATADIR, category) # path to orca or not-orca dir
  for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # rgb data is 3 times the size of grayscale data
    plt.imshow(img_array, cmap="gray")
    plt.show()
    break
  break

# print(img_array.shape)

IMG_SIZE = 150

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = 'gray')
# plt.show()

# print(len(training_data))
# Remember with binary classification, make the data amount between the 2 categories 50/50

# shuffling the data
import random
random.shuffle(training_data)

# for sample in training_data[:10]:
  # print(sample[1])

X = [] # uppercase x (X) is your feature set
y = [] # lowercase y (y) is your labels

for features, label in training_data:
  X.append(features)
  y.append(label)

# We have tp make X a numpy array

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # the 1 is because the img's are grayscale
                                                    # it would be 3 if the img's were RGB (3 values in RGB)

import pickle   # used to save our data

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

y = np.array(y)
X = X/255.0

# Model archictecture
model = Sequential()

# 1st layer
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 3rd layer
model.add(Flatten())
model.add(Dense(64))

# Output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)

import cv2
import tensorflow as tf

CATEGORIES = ["Muscicapa striata", "Tringa ochropus"]

# Use these variables to choose a file that the model will be tested on
bird = 'Muscicapa striata'
file_num = 14

test_file = f'/content/drive/My Drive/AI Club Species Identification Project/Datasets/Test Data (200 files)/{bird} (100 spectrograms)/test{file_num}.png'

def prepare(filepath):
    IMG_SIZE = 150
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap = 'gray')
    plt.show()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

prediction = model.predict([prepare(test_file)])
# print(prediction)
print(f"Prediction: {CATEGORIES[int(prediction[0][0])]}")

if 'Muscicapa striata' in test_file:
  print("Reality:    Muscicapa striata")
elif 'Tringa ochropus' in test_file:
  print("Reality:    Tringa ochropus")

from IPython.display import Audio
import torchaudio

audio_file = f'/content/drive/My Drive/AI Club Species Identification Project/Datasets/Test Data (200 files)/Audio/{bird} (100 audio files)/test{file_num}.wav'
original_audio, sample_rate = torchaudio.load(audio_file)
Audio(data = original_audio, rate = sample_rate)

import cv2
import tensorflow as tf

correct = 0
for file_num in range(0,100):
  test_file = f'/content/drive/My Drive/AI Club Species Identification Project/Datasets/Test Data (200 files)/Muscicapa striata (100 spectrograms)/test{file_num}.png'

  def prepare(filepath):
    IMG_SIZE = 150
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

  prediction = model.predict([prepare(test_file)])
  CATEGORIES[int(prediction[0][0])]
  if CATEGORIES[int(prediction[0][0])] == 'Muscicapa striata':
    correct += 1

for file_num in range(0,100):
  test_file = f'/content/drive/My Drive/AI Club Species Identification Project/Datasets/Test Data (200 files)/Tringa ochropus (100 spectrograms)/test{file_num}.png'

  def prepare(filepath):
    IMG_SIZE = 150
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

  prediction = model.predict([prepare(test_file)])
  CATEGORIES[int(prediction[0][0])]
  if CATEGORIES[int(prediction[0][0])] == 'Tringa ochropus':
    correct += 1
print(correct)

