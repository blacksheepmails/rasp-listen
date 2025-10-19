# tune_listener/__main__.py

import argparse
import librosa
import pyaudio
import numpy as np
import subprocess
from scipy.signal import find_peaks

def int16_to_float32(int16):
    float32 = int16.astype(np.float32)
    float32 /= 32768.0
    return float32

def solfege_to_ratio(s):
    return SCALE[s - 1]

def add_note(note, notes, args):
    avg_note = sum(note[:NOTE_LENGTH]) / NOTE_LENGTH
    notes.append(avg_note)
    debug(f'new note {avg_note}', args)

def debug(m, args):
    if args.debug:
        print(m)

CHUNK = 1024 * 2
RATE = 22050
NOTE_LENGTH = 2
SCALE = [1.0, 9.0 / 8, 5.0 / 4, 4.0 / 3, 3.0 / 2, 5.0 / 3, 15.0 / 8, 2.0]
ERR = 0.06
CLAP_THRESHOLD = 10000
VOLUME_THRESHOLD = 4
NOTE_DRIFT_THRESHOLD = 0.03

def main():
    parser = argparse.ArgumentParser(description='Listen for a short tune, then do a thing.')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debugging output')
    parser.add_argument('--tune', default="123", help='Use solfege to specify a tune, e.g., 1155665 for twinkle twinkle')
    parser.add_argument('--do', default="echo TOOT", help='Shell command to execute when the tune is heard')
    args = parser.parse_args()

    tune = [int(x) for x in list(args.tune)]
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")
    try:
        notes = []
        note = [0]
        while True:
            chunk = stream.read(CHUNK, exception_on_overflow=False)
            data = np.frombuffer(chunk, dtype=np.int16)
            spec = librosa.stft(np.array([int16_to_float32(x) for x in data]))
            single_frame = [max(np.abs(p)) for p in spec]
            peaks, _ = find_peaks(single_frame[:400], prominence=1)

            if len(peaks) == 0:
                if len(note) >= NOTE_LENGTH:
                    add_note(note, notes, args)
                note = [0]
            else:
                f = librosa.fft_frequencies()[peaks[0]]
                if abs(1 - note[-1] / f) < NOTE_DRIFT_THRESHOLD:
                    debug(f'f:{f}, err:{abs(1 - note[-1] / f):.3f}', args)
                    note.append(f)
                else:
                    if len(note) >= NOTE_LENGTH:
                        add_note(note, notes, args)
                    note = [f]

            if len(notes) >= len(tune):
                match_tune = True
                latest_notes = notes[-1 * len(tune):]
                for i in range(1, len(tune)):
                    if abs(latest_notes[i] / latest_notes[0] - solfege_to_ratio(tune[i]) / solfege_to_ratio(tune[0])) > ERR:
                        match_tune = False
                        break
                if match_tune:
                    result_shell = subprocess.run(args.do, shell=True, capture_output=True, text=True)
                    print("Shell Output:", result_shell.stdout)
                    debug('YAY', args)
                    notes = []

            if max(data) > CLAP_THRESHOLD:
                print('clap')

    except KeyboardInterrupt:
        print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()
