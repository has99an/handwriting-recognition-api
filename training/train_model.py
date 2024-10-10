import os
import numpy as np
import tensorflow as tf
from tensorflow import keras  
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger  # Add this import

# Load saved training data from files
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Use data augmentation to create variation in training images
datagen = ImageDataGenerator(
    rotation_range=15,  # Allows rotations of the image up to 15 degrees
    width_shift_range=0.15,  # Shifts the image horizontally by up to 15% of the width
    height_shift_range=0.15,  # Shifts the image vertically by up to 15% of the height
    shear_range=0.15,  # Shears the image along axes
    zoom_range=0.15,  # Zooms in or out of the image by up to 15%
    horizontal_flip=False,  # Disables horizontal flipping (irrelevant for letters and numbers)
    fill_mode='nearest'  # Fills empty areas during distortion with nearest pixel value
)

# Create a generator that delivers batches of images with augmentation
train_generator = datagen.flow(X_train, y_train, batch_size=32)

# Define a convolutional neural network model for image recognition
model = keras.Sequential([
    layers.Input(shape=(32, 32, 1)),  # Input layer accepts 32x32 grayscale images
    layers.Conv2D(64, (3, 3), activation='relu'),  # First convolutional layer with 64 filters
    layers.MaxPooling2D(pool_size=(2, 2)),  # Reduces dimensions by taking max values
    layers.Conv2D(128, (3, 3), activation='relu'),  # Second convolutional layer with 128 filters
    layers.MaxPooling2D(pool_size=(2, 2)),  # Pooling to further reduce image size
    layers.Conv2D(256, (3, 3), activation='relu'),  # Third convolutional layer with 256 filters
    layers.MaxPooling2D(pool_size=(2, 2)),  # Another pooling layer
    layers.Flatten(),  # Flattens the image into a single vector for fully connected layer
    layers.Dense(256, activation='relu'),  # Fully connected layer with 256 neurons
    layers.Dropout(0.5),  # Dropout to prevent overfitting by deactivating 50% of neurons
    layers.Dense(len(np.unique(y_train)), activation='softmax')  # Output layer, one neuron per class
])

# Compile the model with an optimizer, loss function, and evaluation metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Used for multi-class classification
              metrics=['accuracy'])  # Model is evaluated based on accuracy

# Define early stopping to stop training early if the model stops improving
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train the model with the generator providing augmented data and use early stopping
history = model.fit(train_generator, 
                    epochs=100, 
                    callbacks=[early_stopping])

# Save the training history to a file
np.save('C:\\ImageRecognitionOfHandwriting\\api-service\\training\\training_history.npy', history.history)

# Save the trained model for later use
model.save('C:\\ImageRecognitionOfHandwriting\\api-service\\training\\model.keras')  # Saves the model as a .keras file
