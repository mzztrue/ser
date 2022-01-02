# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:35:33 2021

@author: swis
"""
import os

import torch
import torchaudio

from torch.utils.data import Dataset

import tools
from tools import rechannel,resample,pad_trunc,plot_spectrogram,plot_waveform

#labels to emotions
enter_dict = {"an":1, "fe":2, "ha":3, "sa":5, "su":6, "di":7}
emodb_dict = {"W":1, "A":2, "F":3, "N":4, "T":5, "E":7, "L":8}
en_em_list = [1,2,3,5,7]
ca_en_list = [1,2,3,5,6]
ca_em_list = [1,2,3,5]

  #-------------------------------------------------------------------
  #In the common case that you use CrossEntropyLoss 7 as your loss
  #function, per its documentation, your labels (the target) must be
  #integer class labels that run from [0, nClass - 1], inclusive. That
  #is, the labels start from 0.
  #------------------------------------------------------------------------
def label_2_cross_entrophy_class(emo,label_list):
    for idx, label in enumerate (label_list):
        if emo == label:
            return idx 
            


class Audioset(Dataset):
  def __init__(self, root, name_text, relative_aud_dir , labeltype):

    self.aud_dir_prefix = os.path.join(root,relative_aud_dir)
    self.labeltype = labeltype

    self.duration = 3
    self.sr = 16000
    self.channel = 1
    
    self.aud_names = []
    self.aud_labels = []

    namefile = os.path.join(root,name_text)
    with open(namefile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x,y =line.strip().split(" ")
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
    emo = label_2_cross_entrophy_class(emo,self.labeltype)
    label = torch.tensor(emo, dtype = torch.long)
    
    aud  = torchaudio.load(aud_dir)
    waveform, sr = aud[0],aud[1]
    resam = resample(waveform, self.sr)
    rechan = rechannel(resam, self.channel)
    aud = pad_trunc(rechan, self.sr,self.duration)
    mel_spec=tools.mel_spectrogram(aud)

    return mel_spec, label
