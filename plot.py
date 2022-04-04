
import os
import random
import librosa
import numpy as np

import torch
import torchaudio
import matplotlib.pyplot as plt

def rechannel(waveform, new_channel=1):
    """
      take in waveform vector(channel, data) and new channel numbers,
      return rechanneled waveform tuple
    """
    if (waveform.shape[0] == new_channel):
        return waveform

    if (new_channel == 1):
        # Convert from stereo to mono by selecting only the first channel
        new_waveform = waveform[:1, :]
        return new_waveform

def resample(waveform, oldsr, newsr=16000):
  """
  take in waveform vector, oldsr, newsr,
  return resampled waveform tuple
  """
  if (oldsr == newsr):
      return waveform
  # Resample first channel
  new_waveform = torchaudio.transforms.Resample(oldsr, newsr)(waveform[:1, :])
  return new_waveform

def pad_trunc(wf, sr, max_s=3):
  """
  take waveform vector, sample rate * seconds as data point numbers,
  pad or truncate length of waveform vector
  """
  num_rows, wf_len = wf.shape
  max_len = sr * max_s

  if (wf_len > max_len):
      # Truncate the signal to the given length
      begin_idx = random.randint(0, wf_len - max_len)
      end_idx = begin_idx + max_len
      wf = wf[:, begin_idx:end_idx]

  elif (wf_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - wf_len)
      pad_end_len = max_len - wf_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))
      wf = torch.cat((pad_begin, wf, pad_end), 1)
  return wf

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):

    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1) #return (num of channels,1) figures

    if num_channels == 1:
        axes = [axes] #unsqueeze axes

    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}') #f-string format
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)

    figure.suptitle(title)
    plt.show(block=False)

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None,ymax=None):#take a spectrum(...,freq,time)
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    if ymax:
        axs.set_ylim((0, ymax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


DATAROOT ='E:/projects/ser/database'
em_file = "emodb535_raw"
en_file = "enterface1287_raw"
em_text = "emodb2enter.txt"
en_text = "enter2emodb.txt"

namefile = os.path.join(DATAROOT, en_text)

aud_names = []
count = 0
with open(namefile, 'r') as file:
    lines = file.readlines()
    for line in lines:
        count +=1
        x,y = line.strip().split(" ")
        aud_names.append(x)

aud_folder = os.path.join(DATAROOT,en_file)
# aud_dir = os.path.join(aud_folder,aud_names[random.randint(0,count-1)])
aud_dir = os.path.join(aud_folder,aud_names[0])
aud = torchaudio.load(aud_dir)
waveform, old_sample_rate = aud[0], aud[1]

print("load audio:")
plot_waveform(waveform,old_sample_rate,title="original waveform")

sample_rate=16000
channel = 1
duration = 3
waveform = resample(waveform, old_sample_rate, 16000) 
waveform = rechannel(waveform, channel)   
waveform = pad_trunc(waveform, sample_rate, duration)       
#check
print("standardize duration to 3s:")
plot_waveform(waveform,sample_rate,"padded or truncated audio")



n_fft = 1024
win_length = None
hop_length = 512
# define transformation
spectrogram = torchaudio.transforms.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
)

specgram = spectrogram(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))
plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(),origin='lower')
plt.colorbar()

extent = [0 , 3000, 0 , 8000]
fig, ax = plt.subplots(1,1)
img = ax.imshow(librosa.power_to_db(specgram[0,:,:]),origin='lower',extent=extent)
ax.set_xticks([0,1000,2000,3000])
fig.colorbar(img)
# plt.title('How to change imshow axis values with matplotlib ?', fontsize=8)
# plt.savefig("imshow_change_values_on_axis_03.png", bbox_inches='tight')
plt.show()

plot_spectrogram(specgram[0], title=None, ylabel='freq_bin', aspect='auto', xmax=None,ymax=None)#take a spectrum(...,freq,time)


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
    norm="slaney",
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk",
)
mel_specgram = mel_spectrogram(waveform)
print("Shape of mel_spectrogram: {}".format(mel_specgram.size()))

plt.figure()
p = plt.imshow(mel_specgram.log2()[0,:,:].detach().numpy())

mel_specgram = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_specgram)
plt.figure()
p = plt.imshow(mel_specgram[0,:,:].detach().numpy())