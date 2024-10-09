import os
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_images_from_folder(folder):
    images = []
    labels = []
    label_counts = {}
    
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            count = 0
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(subfolder)
                    count += 1
            label_counts[subfolder] = count
            
    return images, labels, label_counts

def preprocess_images(images):
    resized_images = []
    for img in images:
        img = cv.resize(img, (32, 32))
        img = img.astype('float32') / 255.0
        resized_images.append(img)
    return np.array(resized_images)

def prepare_data(image_folder):
    """Indlæser og forbereder billeder fra en given mappe til træning."""
    images, labels, label_counts = load_images_from_folder(image_folder)
    
    for label, count in label_counts.items():
        print(f'{label}: {count} billeder')
    
    print(f'Indlæste {len(images)} billeder i alt.')
    
    images = preprocess_images(images)
    images = np.expand_dims(images, axis=-1)
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    X_train = images
    y_train = labels_encoded
    
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    
    print("Data er blevet gemt som .npy filer.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Brug: python prepare_data.py <sti til billede-mappe>")
        sys.exit(1)
        
    image_folder = sys.argv[1]
    prepare_data(image_folder)
