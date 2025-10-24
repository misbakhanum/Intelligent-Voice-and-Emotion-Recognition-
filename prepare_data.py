import os
import numpy as np
import librosa

DATA_PATH = "speech-emotion-recognition-ravdess-data"
SAVE_PATH = "data/processed"
os.makedirs(SAVE_PATH, exist_ok=True)

X = []
y = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            label = int(file.split("-")[2])
            y.append(label-1)  # 0-indexed

            # Load audio
            signal, sr = librosa.load(path, sr=22050)

            # Fixed librosa >= 0.10 syntax
            mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=64)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Pad or truncate to 128 frames
            if mel_db.shape[1] < 128:
                pad_width = 128 - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
            else:
                mel_db = mel_db[:, :128]

            X.append(mel_db)

X = np.array(X)
y = np.array(y)

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save
np.savez(os.path.join(SAVE_PATH, "ravdess_melspectrograms.npz"),
         X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("Data prepared and saved!")
