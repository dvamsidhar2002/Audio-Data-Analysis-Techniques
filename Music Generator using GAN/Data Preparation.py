import os

# 1. Validate DATA_DIR
DATA_DIR = r'H:\Deep Learning\Music Generator by Genre\GTZAN Data\genres_original'
if not os.path.exists(DATA_DIR):
    raise ValueError(f"The directory {DATA_DIR} does not exist.")
else:
    print(f"Dataset found at {DATA_DIR}")

# 2. Determine SEQUENCE_LENGTH
SAMPLING_RATE = 44100
HOP_LENGTH = 512       # Default librosa hop length
SEQUENCE_DURATION_SEC = 15  # Desired sequence duration in seconds
FRAME_DURATION_MS = (HOP_LENGTH / SAMPLING_RATE) * 1000
SEQUENCE_LENGTH = int(SEQUENCE_DURATION_SEC * 1000 / FRAME_DURATION_MS)
print(f"Sequence Length for {SEQUENCE_DURATION_SEC}s: {SEQUENCE_LENGTH}")