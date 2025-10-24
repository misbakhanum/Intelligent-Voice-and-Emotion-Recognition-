import numpy as np
from keras.models import load_model
import librosa
import IPython.display as ipd  
import os

# Emotion mapping (assuming RAVDESS 8 emotions)
emotion_labels = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

# Load test data and model
data = np.load("data/processed/ravdess_melspectrograms.npz")
X_test = data['X_test']
y_test = data['y_test']
model = load_model("models/speech_emotion_model.h5")

# Reshape for CNN if needed
X_test = X_test[..., np.newaxis]

# Predict first 10 samples
for i in range(10):
    sample = X_test[i:i+1]  # keep batch dimension
    pred = model.predict(sample)
    pred_label = np.argmax(pred)
    actual_label = y_test[i] if y_test.ndim == 1 else np.argmax(y_test[i])
    
    print(f"Sample {i+1}: Actual: {emotion_labels[actual_label]}, Predicted: {emotion_labels[pred_label]}")
    
    # Load original .wav file (update path accordingly)
    wav_path = f"speech-emotion-recognition-ravdess-data\Actor_01.wav"
    if os.path.exists(wav_path):
        audio, sr = librosa.load(wav_path, sr=None)
        ipd.display(ipd.Audio(audio, rate=sr))
    else:
        print("Original audio file not found!")
