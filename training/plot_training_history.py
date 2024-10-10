import numpy as np
import matplotlib.pyplot as plt

# Indlæs træningshistorikken fra filen
history = np.load('C:/ImageRecognitionOfHandwriting/api-service/training/training_history.npy', allow_pickle=True).item()

# Opret figur og akser
plt.figure(figsize=(12, 6))

# Plot træningsnøjagtighed
plt.subplot(2, 1, 1)  # 2 rækker, 1 kolonne, 1. subplot
plt.plot(history['accuracy'], label='Training Accuracy', color='blue', marker='o')
plt.plot(history['loss'], label='Training Loss', color='red', marker='x')
plt.title('Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.grid(True)
plt.legend()

# Plot valideringsnøjagtighed, hvis tilgængelig
if 'val_accuracy' in history:
    plt.subplot(2, 1, 2)  # 2. subplot
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='green', marker='s')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

# Gem grafen
plt.tight_layout()
plt.savefig('C:/ImageRecognitionOfHandwriting/api-service/training/training_history_plot.png')
plt.show()
