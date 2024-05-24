
import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import pyaudio
import wave

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Function to extract MFCC features from an audio file
def extract_mfcc(file_name):
    y, sr = librosa.load(file_name, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

# Streamlit app title
st.title("Speech Emotion Recognition Web App")

# Audio recording and prediction section
st.header("Audio Recorder and Emotion Prediction")

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

# Create a stream object for audio recording
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'ps']
# Button to start recording
if st.button("Start Recording"):
    st.write("Recording...")
    frames = []

    # Record audio for RECORD_SECONDS
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("Recording complete")

    # Save recorded audio to a WAV file
    wf = wave.open("recorded_audio.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    # Extract MFCC features from the recorded audio
    feature = extract_mfcc("recorded_audio.wav")

    # Reshape the feature to match the input shape defined while initializing the model
    feature = feature.reshape(1, 40)

    # Prediction
    speech_emotion = model.predict(feature)

    # Probability Distribution
    emotion_index = np.argmax(speech_emotion)

    # Emotion Labels
    

    # Getting the label of the predicted emotion
    predicted_emotion_label = emotion_labels[emotion_index]

    st.write(f"Predicted Emotion: {predicted_emotion_label}")

# Close the audio stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()

# File uploader section
st.header("Emotion Prediction from Uploaded File")

# Upload an audio file
audio = st.file_uploader("Choose the Audio File (wav or mp3)")

# Button to predict emotion from the uploaded file
if st.button("Predict Emotion"):
    if audio is not None:
        feature = extract_mfcc(audio)

        # Reshape the feature to match the input shape defined while initializing the model
        feature = feature.reshape(1, 40)

        # Prediction
        speech_emotion = model.predict(feature)

        # Probability Distribution
        emotion_index = np.argmax(speech_emotion)

        # Getting the label of the predicted emotion
        predicted_emotion_label = emotion_labels[emotion_index]

        st.write(f"Predicted Emotion: <b>{predicted_emotion_label}</b>" , unsafe_allow_html=True)
