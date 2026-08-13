import sys
import sounddevice as sd, soundfile as sf

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
path = sys.argv[2] if len(sys.argv) > 2 else "corpus/test.wav"
sr = 16000

print(f"recording {dur}s -> {path} ...")
audio = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
sf.write(path, audio, sr)
print(f"saved {path}")
