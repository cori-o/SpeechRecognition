import librosa
import librosa.display
import matplotlib.pyplot as plt

# 백엔드 설정
audio_path = "/ibk/STT/data/output/chunk/chunk_전략기획사업단_01_1.wav"

# Load audio file
y, sr = librosa.load(audio_path)

# Display spectrogram
plt.figure(figsize=(10, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.title("Waveform of the Audio File")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()