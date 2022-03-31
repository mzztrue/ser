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

def time_shift(aud, shift_limit):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

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
        #check
        print("load audio:")
        tools.plot_waveform(waveform,old_sample_rate,title="original waveform")
        
        waveform = tools.resample(waveform, old_sample_rate, self.sample_rate) 
        #check
        print("resample audio:")
        tools.plot_waveform(waveform,self.sample_rate,"resampled audio")
        
        waveform = tools.rechannel(waveform, self.channel)   
        #check
        print("standardize channel to mono:")
        tools.plot_waveform(waveform,self.sample_rate,"rechanneled audio")
        
        waveform = tools.pad_trunc(waveform, self.sample_rate, self.duration)       
        #check
        print("standardize duration to 3s:")
        tools.plot_waveform(waveform,self.sample_rate,"padded or truncated audio")

        mel_spec = tools.mel_spectrogram(waveform)
        #check
        print("Mel Spectrogram of the standardized sample:")
        tools.plt.figure()
        tools.plt.imshow(mel_spec.squeeze().numpy())
        
        print(mel_spec.size())
        mel_spec = torch.transpose(mel_spec,1,2)
        print(mel_spec.size())

        tools.plt.figure()
        tools.plt.imshow(mel_spec.squeeze().numpy())


        mel_spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spec)
        print("Mel Spectrogram in log scale:")
        # tools.plot_spectrogram(mel_spec[0])
        tools.plt.figure()
        tools.plt.imshow(mel_spec.squeeze().numpy())

        

        #do random crop and flip to mel spectrum image, as data augmentation
        preprocess = T.Compose([
            T.RandomResizedCrop((224, 224)),
            # T.RandomHorizontalFlip(),
        ])

        if(self.domaintype=='src'):
            # resized_mel_spec = preprocess(F.resize(mel_spec, (256, 256))).repeat(3, 1, 1)
            resized_mel_spec = preprocess(mel_spec).repeat(3, 1, 1)
        elif(self.domaintype=='tar'):
            resized_mel_spec = F.resize(mel_spec, (224, 224)).repeat(3, 1, 1)
        
        # tools.plt.figure()
        # tools.plt.imshow(resized_mel_spec[0].squeeze())
        #check
        # print("Mel Spectrogram after data augmentation(random crop and flip), resizing(256 by 256) and channel repeating(3 channels):")
        # tools.plot_spectrogram(resized_mel_spec[0])
        # print("shape of the model input",resized_mel_spec.shape)


        return resized_mel_spec, label
