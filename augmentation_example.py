import tensorflow as tf
import matplotlib.pyplot as plt

# Завантаження зображення з файлу (вкажи свій шлях до файлу)
image_path = r"C:\Nauka\diplom work mn cv\content\Training\Training\00021\00375_00000.png"

# Читаємо і декодуємо PNG-зображення
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_png(image_raw, channels=3)  # 3 канали (RGB)

# За потреби змінюємо розмір (наприклад, 224x224)
image = tf.image.resize(image, [224, 224])

# Нормалізуємо пікселі у [0,1]
image = image / 255.0

# Створюємо послідовність аугментацій
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Додаємо batch вимір (batch=1)
image_batch = tf.expand_dims(image, 0)

# Застосовуємо аугментації 3 рази
augmented_images = [image]  # оригінал
for _ in range(3):
    augmented = data_augmentation(image_batch)
    augmented_images.append(tf.squeeze(augmented, axis=0))

# Виводимо оригінал та 3 аугментовані зображення
plt.figure(figsize=(12, 3))
for i, img in enumerate(augmented_images):
    plt.subplot(1, 4, i + 1)
    plt.imshow(img.numpy())
    plt.axis('off')
    if i == 0:
        plt.title('Original')
    else:
        plt.title(f'Augmented {i}')
plt.show()