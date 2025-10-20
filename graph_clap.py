import argparse
import librosa
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import subprocess
import sys
from scipy.signal import find_peaks

def int16_to_float32(int16):
	float32 = int16.astype(np.float32)
	float32 /= 32768.0
	return float32

CLAP_THRESHOLD = 10000
CHUNK = 1024*2  # Number of audio frames per buffer
RATE = 22050  # Sample rate (samples per second)


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, #paInt16 or paUInt8 for lower quality, need to convert to work with librosa
			  channels=1, # Mono
			  rate=RATE,
			  input=True,  # Set to True for input (microphone)
			  frames_per_buffer=CHUNK)


try:
	while True:
		chunk = stream.read(CHUNK, exception_on_overflow = False)
		data = np.frombuffer(chunk, dtype=np.int16)
	  
		abs_data = [abs(x) for x in data]
		if data.max() > CLAP_THRESHOLD:
			i = data.argmax()
			clap = [x > data.max()/2 for x in data]
			plt.plot(clap, alpha=0.5)
			plt.vlines([i], ymin=0, ymax=1, linestyle='dotted', color='red')

			plt.show()
			mean_square = 0
			for j, x in enumerate(clap):
				if x:
					mean_square += abs(i-j)**2
			mean_square /= sum(clap)
			print(mean_square)
			if mean_square < 10000:
				print('clap')
except KeyboardInterrupt:
	stream.stop_stream()
	stream.close()
	p.terminate()
