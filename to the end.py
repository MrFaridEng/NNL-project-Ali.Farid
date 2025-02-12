import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist

# بارگیری مجموعه داده MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # نرمال‌سازی

# تغییر شکل ورودی‌ها برای CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ساخت مدل CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# کامپایل و آموزش مدل
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# ذخیره مدل آموزش‌دیده
model.save("mnist_model.h5")

# نمایش نمودارهای آموزش مدل
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Trend')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Trend')
plt.show()

# بارگیری تصویر کاربر
image_path = "D:/Numberr/New folderrr/test.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = np.invert(image)  # معکوس کردن رنگ‌ها (MNIST پس‌زمینه سیاه دارد)
image = image / 255.0  # نرمال‌سازی
image = image.reshape(1, 28, 28, 1)

# نمایش تصویر ورودی
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title("Input Image")
plt.show()

# بارگیری مدل و پیش‌بینی
model = tf.keras.models.load_model("mnist_model.h5")
prediction = np.argmax(model.predict(image))
print(f"عدد پیش‌بینی‌شده: {prediction}")