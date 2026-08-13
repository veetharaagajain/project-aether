import sys
from faster_whisper import WhisperModel

path = sys.argv[1] if len(sys.argv) > 1 else "corpus/test.wav"

model = WhisperModel("small.en", device="cpu", compute_type="int8")

segments, info = model.transcribe(path, word_timestamps=True, vad_filter=True)

for seg in segments:
    for w in seg.words:
        dur = w.end - w.start
        print(f"{w.start:6.2f} {w.end:6.2f}  {dur:5.2f}  {w.word.strip()}")
