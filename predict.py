import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("waste_model.h5")
class_names = ['glass', 'organic', 'paper']


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class.upper()}\nConfidence: {confidence:.1f}%")
    plt.axis('off')
    plt.show()

    print(f"Predicted: {predicted_class.upper()} ({confidence:.1f}% confidence)")


# Test on an image
predict_image("dataset/test/organic/biological49.jpg")