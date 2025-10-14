import librosa
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
# By default, librosa resamples the audio to 22050 Hz and converts it to mono.
y, sr = librosa.load('clap_test.au', sr=None)
print('Audio loaded with a sampling rate of {} Hz'.format(sr))
rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=1024)[0]
rms_normalized = rms / np.max(rms)

# Set a threshold and find peaks
threshold = 0.5 # Adjust as needed
clap_frames = np.where(rms_normalized > threshold)[0]

# Convert frame indices to time
clap_times = librosa.frames_to_time(clap_frames, sr=sr/2) # this is tied to the frame length of the rms.

print("Detected clap times: {}".format(clap_times))

# Optional: Visualize
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.vlines(clap_times, -1, 1, color='r', linestyle='--', label='Clap Detections')
plt.title('Clap Detection using RMS')
plt.legend()
plt.show()