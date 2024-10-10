import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# Indlæs testdata
X_test = np.load('X_data.npy')  # Opdateret filnavn
y_test = np.load('y_data.npy')  # Opdateret filnavn

# Indlæs den trænede model
model = keras.models.load_model('C:/ImageRecognitionOfHandwriting/api-service/training/model.keras')

# Indlæs LabelEncoder klasserne
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)  # Sørg for, at denne fil eksisterer

# Evaluér modellen med testdata
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy * 100:.2f}%')  # Nøjagtighed i procent

# Forudsig klasser for testdata
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Beregn den samlede nøjagtighed
total_correct = np.sum(predicted_classes == y_test)  # Tæl de korrekte forudsigelser
total_predictions = len(y_test)  # Samlet antal forudsigelser
overall_accuracy = total_correct / total_predictions * 100  # Samlet nøjagtighed i procent

# Sammenlign forudsigelser med de faktiske labels
for i in range(len(predicted_classes)):
    print(f'Actual: {label_encoder.inverse_transform([y_test[i]])[0]}, Predicted: {label_encoder.inverse_transform([predicted_classes[i]])[0]}')

print(f'Overall Accuracy: {overall_accuracy:.2f}%')  # Udskriv samlet nøjagtighed