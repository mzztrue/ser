# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:35:33 2021

@author: swis
"""

import os
import torch
import torchaudio
import torchvision.transforms as T
import torchvision.transforms.functional as F

from torch.utils.data import Dataset

import tools

class Audioset(Dataset):
    '''build the audio dataset to retrieve audio samples'''

    def __init__(self, root, name_text, relative_aud_dir, labeltype, domaintype):

        self.aud_dir_prefix = os.path.join(root, relative_aud_dir)
        self.labeltype = labeltype
        self.domaintype = domaintype

      # -------------------------------------------------------
      # standard audio sample: dur = 3s, sr = 16k, one channel
      # -------------------------------------------------------
        self.duration = 3
        self.sample_rate = 16000
        self.channel = 1
        self.aud_names = []
        self.aud_labels = []

        namefile = os.path.join(root, name_text)
        with open(namefile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                x,y = line.strip().split(" ")
                self.aud_names.append(x)
                self.aud_labels.append(y)

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.aud_labels)

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):

        # Absolute file path of the ith audio file - concatenate aud_dir with aud_names[i]
        aud_dir = os.path.join(self.aud_dir_prefix, self.aud_names[idx])

        # Get the Class ID
        emo = int(self.aud_labels[idx])

        def label_2_cross_entrophy_class(emo, label_list):
            for idx, label in enumerate(label_list):
                if emo == label:
                    return idx

        emo = label_2_cross_entrophy_class(emo, self.labeltype)
        label = torch.tensor(emo, dtype = torch.long)#nn.CrossEntropyLoss expects its label input to be of type torch.Long

        aud = torchaudio.load(aud_dir)
        waveform, old_sample_rate = aud[0], aud[1]
        waveform = tools.resample(waveform, old_sample_rate, self.sample_rate)
        waveform = tools.rechannel(waveform, self.channel)
        waveform = tools.pad_trunc(waveform, self.sample_rate, self.duration)

        mel_spec = tools.mel_spectrogram(waveform)

        mel_spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spec)
        #-------------------------------------------------------------------------------------
        # do random transform only to source domain for training, as data augmentation
        #-------------------------------------------------------------------------------------
        # preprocess = T.Compose([
        #     T.RandomCrop((224, 224)),
        #     T.RandomHorizontalFlip(),
        # ])
        # if(self.domaintype=='src'):
        #     resized_mel_spec = preprocess(F.resize(mel_spec, (256, 256))).repeat(3, 1, 1)
        # elif(self.domaintype=='tar'):
        #     resized_mel_spec = F.resize(mel_spec, (224, 224)).repeat(3, 1, 1)
        #-------------------------------------------------------------------------------------

        #-------------------------------------------------------------------------------------
        # no data augmentation
        resized_mel_spec = F.resize(mel_spec, (224, 224)).repeat(3, 1, 1)
        #-------------------------------------------------------------------------------------
        
        #-------------------------------------------------------------------------------------
        # both augmented
        # preprocess = T.Compose([
        #     T.RandomCrop((224, 224)),
        #     T.RandomHorizontalFlip(),
        # ])
        # resized_mel_spec = preprocess(F.resize(mel_spec, (256, 256))).repeat(3, 1, 1)
        #-------------------------------------------------------------------------------------
        return resized_mel_spec, label
