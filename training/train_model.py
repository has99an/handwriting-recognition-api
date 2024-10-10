import os
import numpy as np
import tensorflow as tf
from tensorflow import keras  
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Indlæs gemte træningsdata fra filer
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Brug data augmentation til at skabe variation i træningsbillederne
datagen = ImageDataGenerator(
    rotation_range=15,  # Tillader rotationer af billedet op til 15 grader
    width_shift_range=0.15,  # Flytter billedet horisontalt med op til 15% af bredden
    height_shift_range=0.15,  # Flytter billedet vertikalt med op til 15% af højden
    shear_range=0.15,  # Forvrænger billedet langs akserne
    zoom_range=0.15,  # Zoomer ind eller ud af billedet med op til 15%
    horizontal_flip=False,  # Slår horisontal spejlvending fra (irrelevant for bogstaver og tal)
    fill_mode='nearest'  # Udfylder tomme områder ved forvrængning med nærmeste pixelværdi
)

# Opret en generator, der leverer batch af billeder med augmentation
train_generator = datagen.flow(X_train, y_train, batch_size=32)

# Definer en konvolutionsbaseret neuralt netværksmodel til billedgenkendelse
model = keras.Sequential([
    layers.Input(shape=(32, 32, 1)),  # Inputlaget accepterer 32x32 gråtonede billeder
    layers.Conv2D(64, (3, 3), activation='relu'),  # Første konvolutionslag med 64 filtre
    layers.MaxPooling2D(pool_size=(2, 2)),  # Reducerer dimensionerne ved at tage max værdier
    layers.Conv2D(128, (3, 3), activation='relu'),  # Andet konvolutionslag med 128 filtre
    layers.MaxPooling2D(pool_size=(2, 2)),  # Pooling for at reducere billedstørrelse yderligere
    layers.Conv2D(256, (3, 3), activation='relu'),  # Tredje konvolutionslag med 256 filtre
    layers.MaxPooling2D(pool_size=(2, 2)),  # Endnu et pooling-lag
    layers.Flatten(),  # Flader billedet ud til en enkelt vektor for fuldt tilsluttet lag
    layers.Dense(256, activation='relu'),  # Fuldt tilsluttet lag med 256 neuroner
    layers.Dropout(0.5),  # Dropout for at forhindre overfitting ved at deaktivere 50% af neuronerne
    layers.Dense(len(np.unique(y_train)), activation='softmax')  # Outputlag, ét neuron pr. klasse
])

# Kompilér modellen med en optimizer, tabsfunktion og evalueringsmetrik
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Bruges til klassifikation af flere klasser
              metrics=['accuracy'])  # Modellen evalueres ud fra nøjagtighed

# Definer early stopping for at stoppe træningen tidligt, hvis modellen holder op med at forbedre sig
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Træn modellen med generatoren, der leverer augmented data, og brug early stopping
model.fit(train_generator, 
          epochs=100, 
          callbacks=[early_stopping])

# Gem den trænede model til senere brug
model.save('C:\\HandwritingRecognition\\model.keras')  # Gemmer modellen som en .keras-fil
