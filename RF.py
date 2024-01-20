import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skimage import io, color, transform
from sklearn import preprocessing
import time
import joblib

import tkinter as tk
from tkinter import filedialog, Label, Button, Text
from PIL import Image, ImageTk  


bg_color = "#F2E5CA" 

dataset_path = "/home/zkerroumi42/ProjetIA-ZNZ/dataset"

class_labels = ["Benign_Keratosis_(BKL)", "Molluscum", "Seborrheic_Keratoses"]

start_time = time.time()


class_info = {
    "Benign_Keratosis_(BKL)": {
        "definition": "Benign keratosis, also known as solar lentigo, is a non-cancerous skin condition caused by sun damage.",
        "medicine": "Topical treatments or cryotherapy may be recommended."
    },
    "Molluscum": {
        "definition": "Molluscum contagiosum is a viral infection that causes small, raised, and painless bumps on the skin.",
        "medicine": "Treatment may involve removal of the lesions through cryotherapy or other procedures."
    },
    "Seborrheic_Keratoses": {
        "definition": "Seborrheic keratosis is a non-cancerous skin tumor that originates from cells called keratinocytes.",
        "medicine": "Typically, no treatment is necessary, but removal may be considered for cosmetic reasons."
    }
}


def load_images_and_labels(folder_path):
    images = []
    labels = []
    for class_label, folder_name in enumerate(class_labels):
        class_folder = os.path.join(folder_path, folder_name)
        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)
            image = io.imread(image_path)
            image = color.rgb2gray(image)  
            image = transform.resize(image, (64, 64))  
            images.append(image.flatten())  
            labels.append(class_label)
    return np.array(images), np.array(labels)


images,labels = load_images_and_labels(dataset_path)


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


n_trees = 100  
min_samples_split = 5 
max_depth = 20  


model_filename = "/home/zkerroumi42/ProjetIA-ZNZ//random_forest_model.joblib"
if os.path.exists(model_filename):
    
    random_forest_classifier = joblib.load(model_filename)
else:
   # ----------------------training -----------------------

    random_forest_classifier = RandomForestClassifier(
        n_estimators=n_trees,
        min_samples_split=min_samples_split,
        max_depth=max_depth
    )
    random_forest_classifier.fit(X_train, y_train)
   
    joblib.dump(random_forest_classifier, model_filename)


end_time = time.time()
training_time_minutes = (end_time - start_time) / 60
print(training_time_minutes)
y_pred_rf = random_forest_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")

window = tk.Tk()
window.title("Illness Predictor")
window.configure(bg=bg_color)  
result_label = Label(window, text="", fg="red", bg=bg_color)
result_label.pack(pady=10)

info_text_area = Text(window, height=10, width=50, bg=bg_color)
info_text_area.pack(pady=10)

accuracy_label = Label(window, text=f"Accuracy: {accuracy_rf:.2%}", bg=bg_color)
accuracy_label.pack(pady=10)
# -------------------------- Tk inter ----------------

canvas = tk.Canvas(window, height=150, width=150, bg=bg_color)
canvas.pack(pady=10)

# ----------------------testing -----------------------
def predict_image():
    file_path = filedialog.askopenfilename()
    my_image = io.imread(file_path)
    my_image = color.rgb2gray(my_image)
    my_image = transform.resize(my_image, (64, 64))
    flattened_image = my_image.flatten()

    standardized_image = scaler.transform([flattened_image])
    your_prediction = random_forest_classifier.predict(standardized_image)
    predicted_class_name = class_labels[your_prediction[0]]

    result_label.config(text=f"Predicted illness is : {predicted_class_name}", fg="red")
    info_text = class_info.get(predicted_class_name, {})
    info_text_area.delete(1.0, tk.END)
    info_text_area.insert(tk.END, f"Definition: {info_text.get('definition', '')}\n\nMedicine: {info_text.get('medicine', '')}")

    image_label = Label(window, text="Your Image:", bg=bg_color)
    image_label.pack(pady=10)

    img = Image.fromarray((my_image * 255).astype(np.uint8))
    img = ImageTk.PhotoImage(img)
    
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img  
    accuracy_rf = accuracy_score([your_prediction], [your_prediction])
    accuracy_label.config(text=f"Accuracy: {accuracy_rf:.2%}", bg=bg_color)

choose_button = Button(window, text="Choose Image", command=predict_image, bg='#76520B')
choose_button.pack(pady=10)
window.mainloop()
dataset_path = "C:/Users/HP/machine_learning_pro/dataset"
class_labels = ["Benign_Keratosis_(BKL)", "Molluscum", "Seborrheic_Keratoses"]
start_time = time.time()

class_info = {
    "Benign_Keratosis_(BKL)": {
        "definition": "Benign keratosis, also known as solar lentigo, is a non-cancerous skin condition caused by sun damage.",
        "medicine": "Topical treatments or cryotherapy may be recommended."
    },
    "Molluscum": {
        "definition": "Molluscum contagiosum is a viral infection that causes small, raised, and painless bumps on the skin.",
        "medicine": "Treatment may involve removal of the lesions through cryotherapy or other procedures."
    },
    "Seborrheic_Keratoses": {
        "definition": "Seborrheic keratosis is a non-cancerous skin tumor that originates from cells called keratinocytes.",
        "medicine": "Typically, no treatment is necessary, but removal may be considered for cosmetic reasons."
    }
}

def load_images_and_labels(folder_path):
    images = []
    labels = []
    for class_label, folder_name in enumerate(class_labels):
        class_folder = os.path.join(folder_path, folder_name)
        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)
            image = io.imread(image_path)
            image = color.rgb2gray(image)  
            image = transform.resize(image, (64, 64))  
            images.append(image.flatten())  
            labels.append(class_label)
    return np.array(images), np.array(labels)

images, labels = load_images_and_labels(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_trees = 100  
min_samples_split = 5 
max_depth = 20  

random_forest_classifier = RandomForestClassifier(
    n_estimators=n_trees,
    min_samples_split=min_samples_split,
    max_depth=max_depth
)
random_forest_classifier.fit(X_train, y_train)

end_time = time.time()
training_time_minutes = (end_time - start_time) / 60
print(f"Training Time: {training_time_minutes} minutes")

y_pred_rf = random_forest_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")