import os
import tkinter as tk
from tkinter import filedialog
from skimage import io, color, transform
from keras.utils import to_categorical
from PIL import Image, ImageTk
from tensorflow import keras
from keras import layers, models
from keras.models import model_from_json
import numpy as np


class_labels = ["1. Eczema 1677", "3. Atopic Dermatitis - 1.25k", "6. Benign Keratosis-like Lesions (BKL) 2624"]
image_height, image_width = 224, 224
num_classes = len(class_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

with open("/home/zkerroumi42/ProjetIA-ZNZ/exemple/exemple/cnn_model_archi.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("/home/zkerroumi42/ProjetIA-ZNZ/exemple/exemple/cnn_model_weights.h5")

def classify_image(image_path):
    image = io.imread(image_path)
    image = color.rgb2gray(image)
    image = transform.resize(image, (image_height, image_width))
    image = image.reshape(1, image_height, image_width, 1)

    prediction = loaded_model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class

def choose_image():
    file_path = filedialog.askopenfilename(title="Choose an Image", filetypes=[("All files", "*.*")])

    img = Image.open(file_path)
    img = img.resize((300, 300), Image.BICUBIC)
    photo = ImageTk.PhotoImage(img)
    panel.config(image=photo)
    panel.image = photo

    result_label.config(text=f"Predicted Disease: {classify_image(file_path)}")

window = tk.Tk()
window.title("Disease Classification")

window.geometry("800x600")
window.configure(bg="grey")

choose_button = tk.Button(window, text="Choose Image", command=choose_image, bg="white")
choose_button.pack(pady=10)

panel = tk.Label(window, bg="grey")
panel.pack()

result_label = tk.Label(window, text="Predicted Disease: ", bg="grey", fg="white")
result_label.pack(pady=10)

window.mainloop()
