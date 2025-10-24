import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed data
data = np.load("data/processed/ravdess_melspectrograms.npz")
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print("Data shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Reshape for CNN: (samples, height, width, channels)
X_train = X_train[..., np.newaxis]  # adds channel dimension
X_test = X_test[..., np.newaxis]

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("One-hot labels shape:", y_train.shape, y_test.shape)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# Evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nAccuracy:", accuracy_score(y_true, y_pred_classes))
print("\nClassification Report:\n", classification_report(y_true, y_pred_classes))

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/speech_emotion_model.h5")
print("Model saved as models/speech_emotion_model.h5")
