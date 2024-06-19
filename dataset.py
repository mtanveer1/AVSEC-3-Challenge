import logging
import os
import random
from os.path import join

import cv2
import numpy as np
import torch
from decord import VideoReader
from decord import cpu
from pytorch_lightning import LightningDataModule
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm import tqdm

from config import *
from utils import subsample_list


class AVSEDataset(Dataset):
    def __init__(self, scenes_root, shuffle=True, seed=SEED, subsample=1,
                 clipped_batch=True, sample_items=True, test_set=False):
        super(AVSEDataset, self).__init__()
        self.test_set = test_set
        self.clipped_batch = clipped_batch
        self.scenes_root = scenes_root
        self.files_list = self.build_files_list
        if shuffle:
            random.seed(SEED)
            random.shuffle(self.files_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
        logging.info("Found {} utterances".format(len(self.files_list)))
        self.data_count = len(self.files_list)
        self.batch_index = 0
        self.total_batches_seen = 0
        self.batch_input = {"noisy": None}
        self.index = 0
        self.max_len = len(self.files_list)
        self.max_cache = 0
        self.seed = seed
        self.window = "hann"
        self.fading = False
        self.sample_items = sample_items

    @property
    def build_files_list(self):
        files_list = []
        for file in os.listdir(self.scenes_root):
            if file.endswith("mixed.wav"):
                files = (join(self.scenes_root, file.replace("mixed", "target")),
                         join(self.scenes_root, file.replace("mixed", "interferer")),
                         join(self.scenes_root, file),
                         join(self.scenes_root, file.replace("_mixed.wav", "_silent.mp4")),
                         )
                if not self.test_set:
                    if all([isfile(f) for f in files]):
                        files_list.append(files)
                else:
                    files_list.append(files)
        return files_list

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        while True:
            try:
                data = {}
                if self.sample_items:
                    clean_file, noise_file, noisy_file, mp4_file = random.sample(self.files_list, 1)[0]
                else:
                    clean_file, noise_file, noisy_file, mp4_file = self.files_list[idx]
                data["noisy_audio"], data["clean"], data["video_frames"] = self.get_data(clean_file, noise_file,
                                                                                         noisy_file, mp4_file)
                data['scene'] = clean_file.replace(self.scenes_root, "").replace("_target.wav", "").replace("/", "")
                return data
            except Exception as e:
                logging.error("Error in loading data: {}".format(e))

    def load_wav(self, wav_path):
        return wavfile.read(wav_path)[1].astype(np.float32) / (2 ** 15)

    def resize_video_frames(self, frames):
        # Create an empty list to store the resized frames
        resized_frames = []

        # Iterate over the frames and resize them
        for frame in frames:
            # Convert the frame to BGR format (OpenCV uses BGR by default)
            bgr_frame = frame[:, :, [2, 1, 0]]

            # Resize the frame using OpenCV's resize function
            resized_frame = cv2.resize(bgr_frame, (img_height, img_width))

            # Append the resized frame to the list
            resized_frames.append(resized_frame)
        return resized_frames
    
    def enhanced_frames(self, frames):
        enhanced_frames = np.zeros_like(frames)  # Initialize an array to store the enhanced frames

        for i in range(len(frames)):
            # Split the RGB frame into individual color channels
            b, g, r = cv2.split(frames[i])

            # Apply histogram equalization to each color channel
            b_eq = cv2.equalizeHist(b)
            g_eq = cv2.equalizeHist(g)
            r_eq = cv2.equalizeHist(r)
        # Merge the equalized color channels back into an RGB frame
        enhanced_frames[i] = cv2.merge([b_eq, g_eq, r_eq])
        
        return enhanced_frames
    
    def get_data(self, clean_file, noise_file, noisy_file, mp4_file):
        noisy = self.load_wav(noisy_file)
        vr = VideoReader(mp4_file, ctx=cpu(0))
        if isfile(clean_file):
            clean = self.load_wav(clean_file)
        else:
            # clean file for test set is not available
            clean = np.zeros(noisy.shape)
        if self.clipped_batch:
            if clean.shape[0] > 48000:
                clip_idx = random.randint(0, clean.shape[0] - 48000)
                video_idx = int((clip_idx / 16000) * 25)
                clean = clean[clip_idx:clip_idx + 48000]
                noisy = noisy[clip_idx:clip_idx + 48000]
            else:
                video_idx = -1
                clean = np.pad(clean, pad_width=[0, 48000 - clean.shape[0]], mode="constant")
                noisy = np.pad(noisy, pad_width=[0, 48000 - noisy.shape[0]], mode="constant")
            if len(vr) < 75:
                #img_height, img_width
                #frames = vr.get_batch(list(range(len(vr))), resize=(img_height, img_width)).asnumpy()
                frames = vr.get_batch(list(range(len(vr)))).asnumpy()
                frames = self.resize_video_frames(frames) 
            else:
                max_idx = min(video_idx + 75, len(vr))
                frames = vr.get_batch(list(range(video_idx, max_idx))).asnumpy()
                frames = self.resize_video_frames(frames) 
            
            frames = self.enhanced_frames(frames)

            bg_frames = np.array(
                [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(len(frames))]).astype(np.float32)
            bg_frames /= 255.0
            if len(bg_frames) < 75:
                bg_frames = np.concatenate(
                    (bg_frames, np.zeros((75 - len(bg_frames), img_height, img_width)).astype(bg_frames.dtype)),
                    axis=0)
        else:
            frames = vr.get_batch(list(range(len(vr)))).asnumpy()
            frames = self.resize_video_frames(frames)

            frames = self.enhanced_frames(frames)

            bg_frames = np.array(
                [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(len(frames))]).astype(np.float32)
            bg_frames /= 255.0
        return noisy, clean, bg_frames[np.newaxis, ...]


class AVSEDataModule(LightningDataModule):
    def __init__(self, batch_size=16):
        super(AVSEDataModule, self).__init__()
        self.train_dataset_batch = AVSEDataset(join(DATA_ROOT, "train/scenes"))
        self.dev_dataset_batch = AVSEDataset(join(DATA_ROOT, "dev/scenes"))
        self.dev_dataset = AVSEDataset(join(DATA_ROOT, "dev/scenes"), clipped_batch=False, sample_items=False)
        self.eval_dataset = AVSEDataset(join(DATA_ROOT, "eval/scenes"), clipped_batch=False, sample_items=False,
                                        test_set=True)
        self.batch_size = batch_size

    def train_dataloader(self):
        assert len(self.train_dataset_batch) > 0, "No training data found"
        return torch.utils.data.DataLoader(self.train_dataset_batch, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        assert len(self.dev_dataset_batch) > 0, "No validation data found"
        return torch.utils.data.DataLoader(self.dev_dataset_batch, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True,
                                           persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=4)


if __name__ == '__main__':

    dataset = AVSEDataModule(batch_size=1).train_dataset_batch
    for i in tqdm(range(len(dataset)), ascii=True):
        data = dataset[i]
        for k, v in data.items():
            print(k, v)
        break