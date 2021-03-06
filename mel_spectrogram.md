## Spectrogram:
Create a spectrogram from a audio signal.

Args:
1. 短时傅里叶：
   
    n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
   
    win_length (int or None, optional): Window size. (Default: ``n_fft``)

    hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
2. 截断补齐
   
    pad (int, optional): Two sided padding of signal. (Default: ``0``)

3. 加窗

    window_fn (Callable[..., Tensor], optional): A function to create a window tensor
        that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)

4. 功率谱
    power (float or None, optional): Exponent for the magnitude spectrogram,
        (must be > 0) e.g., 1 for energy, 2 for power, etc.
        If None, then the complex spectrum is returned instead. (Default: ``2``)

    normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)

    wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)

    center (bool, optional): whether to pad :attr:`waveform` on both sides so
        that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
        (Default: ``True``)

    pad_mode (string, optional): controls the padding method used when
        :attr:`center` is ``True``. (Default: ``"reflect"``)

    onesided (bool, optional): controls whether to return half of results to
        avoid redundancy (Default: ``True``)

    return_complex (bool, optional):
        Deprecated and not used.


    

class Spectrogram(torch.nn.Module):
    
    __constants__ = ["n_fft", "win_length", "hop_length", "pad", "power", "normalized"]

    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        return_complex: Optional[bool] = None,
    ) -> None:
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        if return_complex is not None:
            warnings.warn(
                "`return_complex` argument is now deprecated and is not effective."
                "`torchaudio.transforms.Spectrogram(power=None)` always returns a tensor with "
                "complex dtype. Please remove the argument in the function call."
            )

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).
        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return F.spectrogram(
            waveform,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.power,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
        )

## MelScale

Turn a normal STFT into a mel frequency STFT, using a conversion matrix.  This uses triangular filter banks.

Args:

    n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)

    sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)

    f_min (float, optional): Minimum frequency. (Default: ``0.``)

    f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)

    n_stft (int, optional): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`. (Default: ``201``)

    norm (str or None, optional): If ``'slaney'``, divide the triangular mel weights by the width of the mel band
        (area normalization). (Default: ``None``)

    mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    
    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.

```
class MelScale(torch.nn.Module):
    
    __constants__ = ["n_mels", "sample_rate", "f_min", "f_max"]

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        assert f_min <= self.f_max, "Require f_min: {} < f_max: {}".format(f_min, self.f_max)
        fb = F.melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm, self.mel_scale)
        self.register_buffer("fb", fb)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).
        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        # (..., time, freq) dot (freq, n_mels) -> (..., n_mels, time)
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

        return mel_specgram
```

## Mel_Spectrogram
Create MelSpectrogram for a raw audio signal.

This is a composition of :py:func:`torchaudio.transforms.Spectrogram` and
and :py:func:`torchaudio.transforms.MelScale`.

Sources
    * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
    * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
    * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

Args:

    sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
    n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
    win_length (int or None, optional): Window size. (Default: ``n_fft``)
    hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
    f_min (float, optional): Minimum frequency. (Default: ``0.``)
    f_max (float or None, optional): Maximum frequency. (Default: ``None``)
    pad (int, optional): Two sided padding of signal. (Default: ``0``)
    n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
    window_fn (Callable[..., Tensor], optional): A function to create a window tensor
        that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
    power (float, optional): Exponent for the magnitude spectrogram,
        (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
    normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
    wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
    center (bool, optional): whether to pad :attr:`waveform` on both sides so
        that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
        (Default: ``True``)
    pad_mode (string, optional): controls the padding method used when
        :attr:`center` is ``True``. (Default: ``"reflect"``)
    onesided (bool, optional): controls whether to return half of results to
        avoid redundancy. (Default: ``True``)
    norm (str or None, optional): If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization). (Default: ``None``)
    mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
Example
    >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
    >>> transform = transforms.MelSpectrogram(sample_rate)
    >>> mel_specgram = transform(waveform)  # (channel, n_mels, time)
See also:
    :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
    generate the filter banks.


class MelSpectrogram(torch.nn.Module):
    
    __constants__ = ["sample_rate", "n_fft", "win_length", "hop_length", "pad", "n_mels", "f_min"]

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )
        self.mel_scale = MelScale(
            self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, norm, mel_scale
        )

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).
        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram = self.spectrogram(waveform)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram
