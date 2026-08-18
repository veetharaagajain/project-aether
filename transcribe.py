import sys

import speech
from prosody_core import TRANSCRIBE_LOCALE

path = sys.argv[1] if len(sys.argv) > 1 else "corpus/test.wav"

words, _ = speech.transcribe_file(path, TRANSCRIBE_LOCALE)

for w in words:
    dur = w.end - w.start
    print(f"{w.start:6.2f} {w.end:6.2f}  {dur:5.2f}  {w.word.strip()}")
