import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

model = tf.keras.models.load_model("waste_model.h5")
class_names = ['glass', 'organic', 'paper']

# 10 test images to demo
test_images = [
    "dataset/test/glass/brown-glass1.jpg",
    "dataset/test/glass/brown-glass50.jpg",
    "dataset/test/glass/brown-glass100.jpg",
    "dataset/test/paper/cardboard1.jpg",
    "dataset/test/paper/cardboard50.jpg",
    "dataset/test/paper/cardboard100.jpg",
    "dataset/test/organic/biological49.jpg",
    "dataset/test/organic/biological100.jpg",
    "dataset/test/organic/biological150.jpg",
    "dataset/test/organic/biological200.jpg",
]

print("=" * 50)
print("   WASTE CLASSIFICATION DEMO - GROUP 7")
print("=" * 50)

correct = 0
total = 0

for img_path in test_images:
    if os.path.exists(img_path):
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        pred = model.predict(img_array, verbose=0)
        predicted = class_names[np.argmax(pred[0])]
        confidence = np.max(pred[0]) * 100
        actual = img_path.split("/")[2]

        total += 1
        if predicted == actual:
            correct += 1

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        color = 'green' if predicted == actual else 'red'
        emoji = "✓" if predicted == actual else "✗"
        plt.title(f"{emoji} Actual: {actual}\nPredicted: {predicted} ({confidence:.1f}%)",
                  color=color, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()

        print(f"{emoji} {actual:8s} → {predicted:8s} | {confidence:.1f}%")
    else:
        print(f"Not found: {img_path}")

print("=" * 50)
print(f"   RESULT: {correct}/{total} correct ({correct / total * 100:.0f}%)")
print("=" * 50)