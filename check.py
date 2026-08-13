import soundfile as sf, numpy as np

a, sr = sf.read("corpus/test.wav")
print(f"{len(a)/sr:.1f}s @ {sr}Hz")
print(f"peak {np.abs(a).max():.4f}  rms {np.sqrt((a**2).mean()):.4f}")
