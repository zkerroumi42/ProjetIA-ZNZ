import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage import io, color, transform
from sklearn import preprocessing

dataset_path = "/home/zkerroumi42/Documents/mes cours/machine learning/ProMlearning/dataset"
class_labels = ["Benign_Keratosis_(BKL)", "Molluscum", "Seborrheic_Keratoses"]

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


k_neighbors = 3 
knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)


knn_classifier.fit(X_train, y_train)


y_pred = knn_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


