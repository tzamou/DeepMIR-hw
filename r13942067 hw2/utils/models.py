from typing import Optional, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
import torchaudio
import math
try:
    from filtering import wiener
    from transforms import make_filterbanks, ComplexNorm
except:
    from utils.filtering import wiener
    from utils.transforms import make_filterbanks, ComplexNorm

class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        # print(self.dim, self.dim_conv, self.dim_untouched)
        self.partial_conv3 = nn.Conv1d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        x_init = x
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        # spatial restore
        x = x.permute(0, 2, 1)#rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # print(x.shape)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = x.permute(0, 2, 1)#rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        # print(x.shape)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = x_1.permute(0, 2, 1)# rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x_1 = self.dwconv(x_1)
        x_1 = x_1.permute(0, 2, 1)#rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = x_1 * x_2

        x = self.linear2(x)

        return x + x_init

class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """

    def __init__(
        self,
        nb_bins: int = 4096,
        nb_channels: int = 2,
        hidden_size: int = 512,
        nb_layers: int = 3,
        unidirectional: bool = False,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
        max_bin: Optional[int] = None,
        used_frfn: bool = False
    ):
        super(OpenUnmix, self).__init__()

        self.used_frfn = used_frfn
        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
            if used_frfn:
                self.frfn = FRFN(dim=lstm_hidden_size, hidden_dim=lstm_hidden_size)
        else:
            lstm_hidden_size = hidden_size // 2
            if used_frfn:
                self.frfn = FRFN(dim=lstm_hidden_size * 2, hidden_dim=lstm_hidden_size * 2)

        if not used_frfn:
            self.lstm = LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=nb_layers,
                bidirectional=not unidirectional,
                batch_first=False,
                dropout=0.4 if nb_layers > 1 else 0,
            )


        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

        self.model_name = 'OpenUnmix'

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """

        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        if self.used_frfn:
            x_in = x.permute(1, 0, 2)
            out = self.frfn(x_in).permute(1, 0, 2)
        else:
            # apply 3-layers of stacked LSTM
            # torch.Size([4307, 3, 512])
            out = self.lstm(x)[0]
        # lstm skip connection
        x = torch.cat([x, out], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x.permute(1, 2, 3, 0)


class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """

    def __init__(
        self,
        target_models: Mapping[str, nn.Module],
        niter: int = 0,
        softmask: bool = False,
        residual: bool = False,
        sample_rate: float = 44100.0,
        n_fft: int = 4096,
        n_hop: int = 1024,
        nb_channels: int = 2,
        wiener_win_len: Optional[int] = 300,
        filterbank: str = "torch",
        reconstructed: str = "default"
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

        self.stft, self.istft = make_filterbanks(
            n_fft=n_fft,
            n_hop=n_hop,
            center=True,
            method=filterbank,
            sample_rate=sample_rate,
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        assert reconstructed in ['griffinlim', 'default']
        self.reconstructed = reconstructed

        self.complexnorm = ComplexNorm(mono=nb_channels == 1)

        # registering the targets models
        self.target_models = nn.ModuleDict(target_models)
        # adding till https://github.com/pytorch/pytorch/issues/38963
        self.nb_targets = len(self.target_models)
        # get the sample_rate as the sample_rate of the first model
        # (tacitly assume it's the same for all targets)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

        self.model_name = 'OpenUnmix-Separator'

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """

        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2) #real imag
        # print('audio',audio.shape)
        mix_stft = self.stft(audio)
        # print("mix", mix_stft.shape)
        # (nb_samples, nb_channels, nb_bins, nb_frames)
        # print("mix_phase_copy", mix_phase_copy.shape)
        X = self.complexnorm(mix_stft)
        # print("X",X.shape)
        # initializing spectrograms variable
        # print("spec",spectrograms.shape)

        if self.reconstructed == 'griffinlim':
            griffinlim_wave = torch.zeros((nb_sources,) + audio.shape, dtype=audio.dtype, device=X.device).permute(1, 0, 2, 3)
            for j, (target_name, target_module) in enumerate(self.target_models.items()):
                # apply current model to get the source spectrogram
                target_spectrogram = target_module(X.detach().clone())  # torch.Size([1, 1, 2049, 44])
                # print(target_spectrogram.shape)
                target_spectrogram = target_spectrogram.squeeze(0)
                reconstructed_wave = self.griffinlim_reconstructed(target_spectrogram, length=audio.shape[2])
                griffinlim_wave[:, j, ...] = reconstructed_wave
                # print(griffinlim_wave.shape, 'griffinlim_wave')

            return griffinlim_wave

        elif self.reconstructed == 'default':
            spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)
            for j, (target_name, target_module) in enumerate(self.target_models.items()):
                # apply current model to get the source spectrogram
                target_spectrogram = target_module(X.detach().clone())  # torch.Size([1, 1, 2049, 44])
                spectrograms[..., j] = target_spectrogram
            # transposing it as
            # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
            spectrograms = spectrograms.permute(0, 3, 2, 1, 4)
            # rearranging it into:
            # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
            # into filtering methods
            mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

            # create an additional target if we need to build a residual
            if self.residual:
                # we add an additional target
                nb_sources += 1

            if nb_sources == 1 and self.niter > 0:
                raise Exception(
                    "Cannot use EM if only one target is estimated."
                    "Provide two targets or create an additional "
                    "one with `--residual`"
                )
            # estimate phase
            nb_frames = spectrograms.shape[1]
            targets_stft = torch.zeros(mix_stft.shape + (nb_sources, ), dtype=audio.dtype, device=mix_stft.device)
            # print('targets_stft', targets_stft.shape)
            for sample in range(nb_samples):
                pos = 0
                if self.wiener_win_len:
                    wiener_win_len = self.wiener_win_len
                else:
                    wiener_win_len = nb_frames
                while pos < nb_frames:
                    cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                    pos = int(cur_frame[-1]) + 1

                    targets_stft[sample, cur_frame] = wiener(
                        spectrograms[sample, cur_frame],
                        mix_stft[sample, cur_frame],
                        self.niter,
                        softmask=self.softmask,
                        residual=self.residual,
                    )

            # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
            targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()
            # print('targets_stft', targets_stft.shape)

            # inverse STFT
            estimates = self.istft(targets_stft, length=audio.shape[2])
            # print(estimates.shape)
            return estimates

    def griffinlim_reconstructed(self, spectrogram, length):
        # spectrogram's shape = (n_samples, f, t)

        transform = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, hop_length=self.n_hop, length=length, n_iter=32, power=1)
        if transform.window is None:
            transform.window = nn.Parameter(torch.hann_window(self.n_fft), requires_grad=False).to(spectrogram.device)
        else:
            transform.window = transform.window.to(spectrogram.device)
        waveform = transform(spectrogram)
        return waveform.unsqueeze(0).unsqueeze(0)

    def to_dict(self, estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        # in the case of residual, we added another source
        if self.residual:
            estimates_dict["residual"] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict

if __name__ == '__main__':
    target_model = OpenUnmix(nb_bins=2049, nb_channels=1)
    model = Separator(target_models={'vocals': target_model}, reconstructed='griffinlim')
    audio = torch.zeros((3, 1, 4410000)) #batch, channel, time
    y = model(audio)
