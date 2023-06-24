import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tifffile

# Menentukan jalur folder dataset
dataset_path = os.path.join(os.getcwd(), 'dataset')  # Menggunakan jalur absolut

# Menentukan ukuran target yang diinginkan
target_size = (24, 24)

# Menentukan ukuran batch
batch_size = 32

# Fungsi untuk memuat gambar dengan ekstensi .tif dan mengubahnya menjadi array numpy
def load_tiff_image(image_path, target_size):
    image = tifffile.imread(image_path)
    image_resized = tf.image.resize(image, target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    return image_array

# Membuat objek ImageDataGenerator untuk melakukan augmentasi atau pemrosesan tambahan
datagen = ImageDataGenerator(rescale=1./255)  # Contoh augmentasi: normalisasi skala piksel

# Memuat data pelatihan dari folder 'dataset1' menggunakan objek ImageDataGenerator
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),  # Ubah sesuai dengan ukuran yang diinginkan
    batch_size=batch_size
)

# Lakukan pra-pemrosesan dan ubah ukuran gambar menjadi 24x24 pixel
resized_train_data = []
for images, labels in train_data:
    resized_images = []
    for image_path in images:
        tiff_image = load_tiff_image(image_path, target_size)
        resized_images.append(tiff_image)
    resized_images = tf.stack(resized_images)
    resized_train_data.append((resized_images, labels))
    
    if len(resized_train_data) >= len(train_data):
        break

# Pastikan bahwa direktori 'dataset1' berada di direktori root proyek Anda sebelum menjalankan main.py.
