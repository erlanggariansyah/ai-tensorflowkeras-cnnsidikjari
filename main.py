import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tifffile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

dataset_path = os.path.join(os.getcwd(), 'dataset')
target_size = (24, 24)
batch_size = 40

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
result = model.fit(resized_train_data[0][0], resized_train_data[0][1], epochs=15)

train_loss = result.history['loss']
train_accuracy = result.history['accuracy']

## MELAKUKAN PENGUJIAN HUBUNGAN NILAI LOSS DAN AKURASI DENGAN EPOCH
epochs = range(1, len(train_loss) + 1)

# Membuat grafik loss
plt.plot(epochs, train_loss, 'r', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Membuat grafik akurasi
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

## MELAKUKAN PENGUJIAN DENGAN DATA TESTING
datatest_path = os.path.join(os.getcwd(), 'datatest')
test_class_names = sorted(os.listdir(datatest_path))

test_data = datagen.flow_from_directory(
    datatest_path,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    classes=test_class_names,
    shuffle=False
)

predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

test_result_class_labels = [class_names[i] for i in predicted_classes]
true_labels = test_data.classes

for i in range(len(test_result_class_labels)):
    print(f"Gambar ke-{i+1}: Prediksi = {test_result_class_labels[i]}, Sebenarnya = {class_names[true_labels[i]]}")
