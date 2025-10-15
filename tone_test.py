import librosa
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import sys

def int16_to_float32(int16):
    float32 = int16.astype(np.float32)
    float32 /= 32768.0
    return float32

def solfege_to_ratio(s):
  return SCALE[s-1]

CHUNK = 1024*2  # Number of audio frames per buffer
RATE = 22050  # Sample rate (samples per second)
NOTE_LENGTH = 2
SCALE = [1.0, 9.0/8, 5.0/4, 4.0/3, 3.0/2, 5.0/3, 15.0/8, 2.0] # Just intonation of major scale
ERR = 0.06
CLAP_THRESHOLD = 10000
VOLUME_THRESHOLD = 4
NOTE_DRIFT_THRESHOLD = 0.03

if __name__ == "__main__":
  tune = [1,3,5] # tune in solfege
  if len(sys.argv) > 1:
    tune = [int(x) for x in list(sys.argv[1])]
    print(tune)
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paInt16, #paInt16 or paUInt8 for lower quality, need to convert to work with librosa
                channels=1, # Mono
                rate=RATE,
                input=True,  # Set to True for input (microphone)
                frames_per_buffer=CHUNK)

  print("Recording...")
  try:
    notes = []
    note = [0]
    while True:
        chunk = stream.read(CHUNK, exception_on_overflow = False)
        data = np.frombuffer(chunk, dtype=np.int16)
        spec = librosa.stft(np.array([int16_to_float32(x) for x in data]))
        single_frame = [max(np.abs(p)) for p in spec]
        if max(single_frame) < VOLUME_THRESHOLD:
          if len(note) >= NOTE_LENGTH:
              notes.append(sum(note[:NOTE_LENGTH])/NOTE_LENGTH) # note is average over most recent
              print(f'new note {sum(note[:NOTE_LENGTH])/NOTE_LENGTH}')
          note = [0]
        else:
          i = single_frame.index(max(single_frame))
          f = librosa.fft_frequencies()[i]
          if abs(1 - note[-1]/f) < NOTE_DRIFT_THRESHOLD:
            print(f'f:{f}, err:{abs(1 - note[-1]/f):.3f}')
            note.append(f)
          else:
            if len(note) >= NOTE_LENGTH:
              notes.append(sum(note[:NOTE_LENGTH])/NOTE_LENGTH) # note is average over most recent
              print(f'new note {sum(note[:NOTE_LENGTH])/NOTE_LENGTH}')
            note = [f]
        if len(notes) >= len(tune):
          match_tune = True
          latest_notes = notes[-1*len(tune):]
          for i in range(1, len(tune)):
            if abs(latest_notes[i]/latest_notes[0] - solfege_to_ratio(tune[i])/solfege_to_ratio(tune[0])) > ERR:
              match_tune = False
              break
          if match_tune == True:
            print('YAY')
            notes = []


        # plt.plot(single_frame[:10000])
        # plt.show()
        # fig, ax = plt.subplots()
        # img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec), ref=np.max), y_axis='log', x_axis='time', ax=ax)
        # ax.set_title('Power spectrogram')
        # fig.colorbar(img, ax=ax, format="%+2.0f dB")
        # plt.show()

        if max(data) > CLAP_THRESHOLD:
           print('clap')
  except KeyboardInterrupt:
    print("Recording stopped.")

  stream.stop_stream()
  stream.close()
  p.terminate()