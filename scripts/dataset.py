import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

import pickle

import numpy as np

class AudioDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.wav_paths = f.read().split('\n')[:-1]
            """
            self.wav_paths = []
            for wp in f.read().split('\n')[:-1]:
                if "s0019" in wp:
                    self.wav_paths.append(wp)
            """
        self.seglist = []
        for wav_path in self.wav_paths:
            s, e = wav_path.split("/")[-1].split("_")[-1].split(".")[0].split("-")
            self.seglist.append((0, int(e)-int(s)))
    
    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, id):
        file_path = self.wav_paths[id]
        #audio, _ = torchaudio.load(file_path)

        return file_path, self.seglist[id]
