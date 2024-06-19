from os.path import isfile

SEED = 1143  # Random seed for reproducibility
sampling_rate = 16000  # Sampling rate for audio
max_frames = 75  # Maximum number of frames per video for training
max_audio_len = sampling_rate * 3  # Maximum number of audio samples per video for training
img_height, img_width = 128, 128  # Image height and width for training

DATA_ROOT = "E:/AVSE_Data/tmps/"
assert not isfile(DATA_ROOT), "Please set DATA_ROOT in config.py to the correct path to the avsec dataset"
