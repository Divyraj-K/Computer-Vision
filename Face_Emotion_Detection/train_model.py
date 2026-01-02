import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

# Load dataset
data = pd.read_csv("dataset/fer2013.csv")

pixels = data['pixels'].tolist()
faces = np.array([np.fromstring(pixel, sep=' ') for pixel in pixels])
faces = faces.reshape(-1, 48, 48, 1)
faces = faces / 255.0

labels = to_categorical(data['emotion'], num_classes=7)

# CNN Model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(faces, labels, epochs=15, batch_size=64)

model.save("model/emotion_model.h5")
print("Model saved successfully!")
