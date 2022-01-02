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
def resample(waveform, sr, newsr=16000):
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

#----------------------------------------------
#1.make frames
# Typical frame sizes in speech processing range from 20 ms to 40 ms with 50% (+/-10%) overlap between consecutive frames. 
# Popular settings are 25 ms for the frame size, 
#
# frame_size = 0.025 and a 10 ms stride (15 ms overlap), frame_stride = 0.001.
#wind_length = 0.025*16000=400,hop_length = 0.01*16000=160
#
#2.hamming window
#3.nfft
#We can now do an N-point FFT on each frame to calculate the frequency spectrum, 
# which is also called Short-Time Fourier-Transform (STFT), where N is typically 256 or 512, NFFT = 512; 
# and then compute the power spectrum (periodogram) 
#4.mel filter bank
# The final step to computing filter banks is applying triangular filters, 
# typically 40 filters, nfilt = 40 on a Mel-scale to the power spectrum to extract frequency bands. 
# The Mel-scale aims to mimic the non-linear human ear perception of sound, 
# by being more discriminative at lower frequencies and less discriminative at higher frequencies. 
#
#
#
n_fft = 1024
win_length = n_fft
hop_length = 512
n_mels = 128

# n_fft = 512
# win_length = n_fft
# hop_length = 256
# n_mels = 40

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
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
