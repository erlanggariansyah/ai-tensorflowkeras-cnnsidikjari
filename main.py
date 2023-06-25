import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tifffile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

dataset_path = os.path.join(os.getcwd(), 'dataset')
target_size = (24, 24)
batch_size = 32

def load_tiff_image(image_path, target_size):
    image = tifffile.imread(image_path)
    image = tf.expand_dims(image, axis=-1)
    image_resized = tf.image.resize(image, target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    return image_array

datagen = ImageDataGenerator(rescale=1./255)

class_names = sorted(os.listdir(dataset_path))

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    classes=class_names,
    shuffle=True
)

resized_train_data = []
for images, labels in train_data:
    resized_images = []
    for i in range(len(images)):
        image_path = train_data.filepaths[i]
        tiff_image = load_tiff_image(image_path, target_size)
        resized_images.append(tiff_image)
    resized_images = tf.stack(resized_images)
    resized_train_data.append((resized_images, labels))
    
    if len(resized_train_data) >= len(train_data):
        break

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(resized_train_data[0][0], resized_train_data[0][1], epochs=10)
