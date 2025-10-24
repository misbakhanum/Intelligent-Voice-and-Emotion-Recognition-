import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from keras.models import load_model

# Constants
FS = 22050          # Sampling rate
DURATION = 10       # seconds
MODEL_PATH = "models/speech_emotion_model.h5"

# Load trained model
model = load_model(MODEL_PATH)

# Emotion labels (example mapping, adjust based on training)
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
RISK_EMOTIONS = ['fear', 'disgust', 'angry']  # considered risky

def record_audio(filename="my_voice.wav"):
    print(f"Recording for {DURATION} seconds...")
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
    sd.wait()
    sf.write(filename, recording, FS)
    print(f"Saved recording as {filename}")
    return filename

def preprocess_audio(file_path):
    signal, sr = librosa.load(file_path, sr=FS)

    # Ensure exactly 10 seconds
    max_len = DURATION * FS
    if len(signal) < max_len:
        signal = np.pad(signal, (0, max_len - len(signal)))
    else:
        signal = signal[:max_len]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize width to 128 (like training)
    if mel_db.shape[1] < 128:
        mel_db = np.pad(mel_db, ((0,0),(0,128 - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :128]

    # Add channel and batch dimension for CNN
    mel_db = mel_db[np.newaxis, ..., np.newaxis]  # shape: (1, 64, 128, 1)
    return mel_db

def predict_emotion(file_path):
    mel_input = preprocess_audio(file_path)
    pred = model.predict(mel_input)
    pred_class_idx = np.argmax(pred, axis=1)[0]
    pred_emotion = EMOTIONS[pred_class_idx]
    status = "RISK" if pred_emotion in RISK_EMOTIONS else "NORMAL"
    return pred_emotion, status

if __name__ == "__main__":
    audio_file = record_audio()
    emotion_class, status = predict_emotion(audio_file)
    print(f"Predicted Emotion: {emotion_class}")
    print(f"Status: {status}")
