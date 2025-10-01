import librosa as lib
import numpy as np

# load audio file
audioFile = input("Please type the audio file name and extension: ")
y, sr = lib.load(audioFile)

# Chroma Short-Time Fourier Transform
chromagram = lib.feature.chromma_stft(y = y, sr = sr)

# Mean chroma feature across time
averageChroma = np.mean(chromagram, axis = 1)

# Mapping chroma features to keys

keyMap = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Select max chroma feature
keyEstimate = keyMap[np.argmax(averageChroma)]

# Print key
print("Esitmated Key:", keyEstimate)