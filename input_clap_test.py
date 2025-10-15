import pyaudio
import numpy as np

CHUNK = 1024*2  # Number of audio frames per buffer
RATE = 22050  # Sample rate (samples per second)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, #paInt16 or paUInt8 for lower quality, need to convert to work with librosa
              channels=1, # Mono
              rate=RATE,
              input=True,  # Set to True for input (microphone)
              frames_per_buffer=CHUNK)

print("Recording...")
try:
  while True:
      data = stream.read(CHUNK)
      audio_data = np.frombuffer(data, dtype=np.float32)
      #rms = librosa.feature.rms(y=audio_data, frame_length=1024, hop_length=1024)
      if max(audio_data) > 1:
         print('clap')
except KeyboardInterrupt:
  print("Recording stopped.")

stream.stop_stream()
stream.close()
p.terminate()