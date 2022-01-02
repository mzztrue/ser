# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 00:01:16 2021
语音转梅尔频谱图
@author: swis
"""

import os
import random
import librosa

import torch
import torchaudio


import matplotlib.pyplot as plt

#plot tools
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)
    
def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

#set to same channel
def rechannel(waveform, new_channel=1):

    if (waveform.shape[0] == new_channel):
      return waveform

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      new_waveform = waveform[:1, :]
      return new_waveform
    
#standardize sample rate
def resample(waveform, sr, newsr):
  if (sr == newsr):
    return waveform
  # Resample first channel
  new_waveform = torchaudio.transforms.Resample(sr, newsr)(waveform[:1,:])
  return new_waveform

#standardize length
def pad_trunc(wf, sr, max_s):
    num_rows, wf_len = wf.shape    
    max_len = sr * max_s

    if (wf_len > max_len):
      # Truncate the signal to the given length
      wf = wf[:,:max_len]

    elif (wf_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - wf_len)
      pad_end_len = max_len - wf_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))
      wf = torch.cat((pad_begin, wf, pad_end), 1)
    return wf

    
#%%
#test audio process

#get dirs:
root = "C:/Users/swis/Desktop/cc-SER/database"

em_list = "emodb2enter.txt"
en_list = "enter2emodb.txt"
em_folder = "emodb535_raw"
en_folder = "enterface1287_raw"

#get enlish audio names
en_aud_names = []

name_list = os.path.join(root,en_list)

with open(name_list, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        x,y =line.strip().split(" ")
        en_aud_names.append(x)

#get aud dir :(root_dir+name), and process audio
for idx, aud_name in enumerate(en_aud_names):
    en_file_path = os.path.join(root, en_folder, aud_name)
    aud = torchaudio.load(en_file_path)
    waveform, sr = aud[0],aud[1]

    #rechannel
    waveform = rechannel(waveform)

    #resample
    sample_rate=16000
    waveform = resample(waveform, sr, sample_rate)

    #pad and trunc
    max_s =3
    waveform = pad_trunc(waveform, sample_rate,max_s)

    #mel spectrogram
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(waveform)

    if(idx %100==0):
        plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')
        plt.show()
        print(melspec.shape)



# %%
