import os
import cv2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from skimage import io, color, transform


#image_height, image_width, num_channels = 224, 224, 3
#num_classes = len(os.listdir(dataset_path))


dataset_path = "D:\Studies\Projet_Machine_learning\Machine learning\Machine learning\IMG_CLASSES"
class_labels = ["1. Eczema 1677", "3. Atopic Dermatitis - 1.25k", "6. Benign Keratosis-like Lesions (BKL) 2624"]


def load_images_and_labels(folder_path):
    images = []
    labels = []

    for class_label, folder_name in enumerate(class_labels):
        class_folder = os.path.join(folder_path, folder_name)
        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)
            image = io.imread(image_path)
            image = color.rgb2gray(image)
            image = transform.resize(image, (224, 224))
            images.append(image.flatten())
            labels.append(class_label)
    return np.array(images), np.array(labels)

images, labels = load_images_and_labels(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("Confusion Matrix:")
confusion_matrix = np.zeros((len(np.unique(labels)), len(np.unique(labels))))
for i in range(len(y_test)):
    confusion_matrix[y_test[i]][y_pred[i]] += 1

print(confusion_matrix)
