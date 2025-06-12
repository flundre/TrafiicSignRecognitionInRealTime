import numpy as np
import os 
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

model = load_model('traffic_sign_model.keras')
cwd = os.getcwd()
class_names = [f"{i:05d}" for i in range(62)]

img_width, img_height = 64, 64

def predict_image(image_path):
    img = Image.open(image_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_resized = img.resize((img_width, img_height))

    img_array = np.array(img_resized)

    img_array = np.expand_dims(img_array, axis=0)

    pred_probs = model.predict(img_array)
    pred_index = np.argmax(pred_probs, axis=1)
    pred_class = class_names[pred_index[0]]

    plt.imshow(img_resized)
    plt.title(f"Predicted Class: {pred_class}")
    plt.axis('off')
    plt.show()

    return pred_class

image_path = cwd + "/content/Figure_1.png"
print(f"Predicted Class: {predict_image(image_path)}")