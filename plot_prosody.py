import sys
import numpy as np
import parselmouth
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import speech
from prosody_core import TRANSCRIBE_LOCALE

path = sys.argv[1] if len(sys.argv) > 1 else "corpus/test.wav"
out = sys.argv[2] if len(sys.argv) > 2 else "prosody.png"

words, _ = speech.transcribe_file(path, TRANSCRIBE_LOCALE)

snd = parselmouth.Sound(path)
total = snd.get_total_duration()

pitch = snd.to_pitch()
pitch_t = pitch.xs()
pitch_hz = pitch.selected_array['frequency'].copy()
pitch_hz[pitch_hz == 0] = np.nan  # break the line at unvoiced frames

intensity = snd.to_intensity()
int_t = intensity.xs()
int_db = intensity.values[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

ax1.plot(pitch_t, pitch_hz, linewidth=1.2, color="tab:blue")
ax1.set_ylabel("pitch (Hz)")

ax2.plot(int_t, int_db, linewidth=1.2, color="tab:red")
ax2.set_ylabel("intensity (dB)")
ax2.set_xlabel("time (s)")

for ax in (ax1, ax2):
    for w in words:
        ax.axvline(w.start, color="0.7", linewidth=0.6)
        ax.axvline(w.end, color="0.7", linewidth=0.6)
    ax.set_xlim(0, total)

# praat reports -300 dB in true digital silence; ignore those frames for the y-range
vis = int_db[int_db > -200]
ymin, ymax = (vis.min(), vis.max()) if len(vis) else ax2.get_ylim()
ax2.set_ylim(ymin - 0.25 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
label_y = ax2.get_ylim()[0] + 0.02 * (ymax - ymin)
for w in words:
    ax2.text((w.start + w.end) / 2, label_y, w.word.strip(),
             ha="center", va="bottom", fontsize=8, rotation=45)

fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"saved {out}")
