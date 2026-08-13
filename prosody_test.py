import sys
import numpy as np
import parselmouth
from faster_whisper import WhisperModel

path = sys.argv[1] if len(sys.argv) > 1 else "corpus/test.wav"

model = WhisperModel("small.en", device="cpu", compute_type="int8")
segments, info = model.transcribe(path, word_timestamps=True, vad_filter=True)

words = [w for seg in segments for w in seg.words]

snd = parselmouth.Sound(path)
total = snd.get_total_duration()

pitch = snd.to_pitch()
pitch_t = pitch.xs()
pitch_hz = pitch.selected_array['frequency']  # 0.0 marks unvoiced frames

intensity = snd.to_intensity()
int_t = intensity.xs()
int_db = intensity.values[0]


def mean_pitch(start, end):
    m = (pitch_t >= start) & (pitch_t <= end)
    v = pitch_hz[m]
    v = v[(v > 0) & ~np.isnan(v)]
    return float(v.mean()) if len(v) else float('nan')


def mean_intensity(start, end):
    m = (int_t >= start) & (int_t <= end)
    v = int_db[m]
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else float('nan')


print(f"{'word':<14}{'start':>7}{'end':>7}{'dur':>7}{'pitch_Hz':>10}{'int_dB':>9}{'gap_before':>12}{'gap_after':>11}")
for i, w in enumerate(words):
    dur = w.end - w.start
    gap_before = w.start - (words[i - 1].end if i > 0 else 0.0)
    gap_after = (words[i + 1].start if i + 1 < len(words) else total) - w.end
    p = mean_pitch(w.start, w.end)
    d = mean_intensity(w.start, w.end)
    print(f"{w.word.strip():<14}{w.start:7.2f}{w.end:7.2f}{dur:7.2f}"
          f"{p:10.1f}{d:9.1f}{gap_before:12.2f}{gap_after:11.2f}")
