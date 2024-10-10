import os
import cv2 as cv
import numpy as np
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
    images, labels, label_counts = load_images_from_folder(image_folder)
    
    for label, count in label_counts.items():
        print(f'{label}: {count} billeder')
    
    print(f'Indlæste {len(images)} billeder i alt.')
    
    images = preprocess_images(images)
    images = np.expand_dims(images, axis=-1)
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Gem data direkte i den eksisterende training-mappe
    output_folder = os.getcwd()  # Gå til nuværende arbejdsmappe (training)
    np.save(os.path.join(output_folder, 'X_data.npy'), images)
    np.save(os.path.join(output_folder, 'y_data.npy'), labels_encoded)

    # Gem label_encoder klasserne
    np.save(os.path.join(output_folder, 'label_classes.npy'), label_encoder.classes_)
    
    print("Data er blevet gemt som .npy filer.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        print("Brug: python prepare_data.py <train eller test>")
        sys.exit(1)

    if sys.argv[1] == 'train':
        image_folder = '../data/train_images'
    else:
        image_folder = '../data/test_images'
    
    prepare_data(image_folder)
