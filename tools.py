import random
import librosa
import numpy as np
import itertools

import torch
import torchaudio
import matplotlib.pyplot as plt

#-----------------------------------------------------
# turn into standard sample(default: 1 channel, 16k sr, 3 sec)
#-----------------------------------------------------

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


# ----------------------------------------------
# 1.make frames
# Typical frame sizes in speech processing range from 20 ms to 40 ms 
# with 50% (+/-10%) overlap between consecutive frames.
# Popular settings are 25 ms for the frame size, and a 10 ms stride (15 ms overlap),
# frame_size = 0.025  frame_stride = 0.001.
# wind_length = 0.025*16000=400, hop_length = 0.01*16000=160

# 2.hamming window
# add hamming window filter

# 3.nfft
# do an N-point FFT on each frame to calculate the frequency spectrum,and then compute the power spectrum (periodogram)
# where N is typically 256 or 512, 
# NFFT = 512

# 4.mel filter bank
# The final step to computing filter banks is applying triangular filters,
# typically 40 filters, nfilt = 40 on a Mel-scale to the power spectrum to extract frequency bands.
# The Mel-scale aims to mimic the non-linear human ear perception of sound,
# by being more discriminative at lower frequencies and less discriminative at higher frequencies.

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

#---------------------
# plot tools
#---------------------
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):

    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1) #return squeezed axes, in this case, num of channels

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

def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix', figsize=[6.4, 4.8], cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    figure = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def get_all_labels_preds(model, loader):
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    for batch in loader:
        images, labels = batch
        preds = model(images)
        all_labels = torch.cat((all_labels, labels), dim=0)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_labels, all_preds


def get_uar(cm):
    tp = []
    fn = []
    uar = []
    for i in range(cm.shape[0]):
        false = 0
        for j in range(cm.shape[1]):
            if i == j:
                tp.append(cm[i][j])
            else:
                false += cm[i][j]
        fn.append(false)
        uar.append(tp[-1]/(tp[-1]+fn[-1]))
    uar = sum(uar)/len(uar)
    return uar


def add_confusion_matrix(
    writer,
    cmtx,
    num_classes,
    global_step=None,
    subset_ids=None,
    class_names=None,
    tag="Confusion Matrix",
    figsize=None,
):
    """
    Calculate and plot confusion matrix to a SummaryWriter.
    Args:
        writer (SummaryWriter): the SummaryWriter to write the matrix to.
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of class_names.
        global_step (Optional[int]): current step.
        subset_ids (list of ints): a list of label indices to keep.
        class_names (list of strs, optional): a list of all class names.
        tag (str or list of strs): name(s) of the confusion matrix image.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    """
    if subset_ids is None or len(subset_ids) != 0:
        # If class names are not provided, use class indices as class names.
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        # If subset is not provided, take every class_names.
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = plot_confusion_matrix(
            sub_cmtx,
            class_names=sub_names,
            normalize=False,
            title=tag,
            figsize=figsize
        )
        # Add the confusion matrix image to writer.
        writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)
