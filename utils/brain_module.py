from functools import partial
import random
import typing as tp
import math
import torch
from torch import nn
from torch.nn import functional as F
# import torchaudio as ta
import numpy as np
import torch.nn.init as init

class ClipLoss(torch.nn.Module):
    """CLIP (See Open AI CLIP) constrastive loss.
    """
    def __init__(self, linear=None, twin=True, pool=False, tmin=None, tmax=None,
                 tmin_train=None, tmax_train=None, center=False):
        super().__init__()
        self.linear = None
        self.pool = pool
        self.center = center
        if linear is not None:
            self.linear_est = torch.nn.LazyLinear(linear)
            if twin:
                self.linear_gt = self.linear_est
            else:
                self.linear_gt = torch.nn.LazyLinear(linear)
        self.tmin = tmin
        self.tmax = tmax
        self.tmin_train = tmin_train
        self.tmax_train = tmax_train

    def trim_samples(self, estimates, candidates):
        """Given estimates that is [B1, C, T] and candidates
        which is [B2, C, T], return estimates_trim of size [B1, C, T']
        and candidates_trim of size [B2, C, T'], such that T'
        corresponds to the samples between [self.tmin, self.tmax]
        """
        trim_min, trim_max = self.tmin, self.tmax
        if trim_min is None:
            trim_min = 0
        if trim_max is None:
            trim_max = estimates.shape[-1]
        estimates_trim = estimates[..., trim_min:trim_max]
        candidates_trim = candidates[..., trim_min:trim_max]
        return estimates_trim, candidates_trim

    def get_scores(self, estimates: torch.Tensor, candidates: torch.Tensor):
        """Given estimates that is [B, C, T] and candidates
        which is [B', C, T], return a [B, B'] matrix of scores of matching.
        """
        estimates, candidates = self.trim_samples(estimates, candidates)
        if self.linear:
            estimates = self.linear_est(estimates)
            candidates = self.linear_gt(candidates)
        if self.pool:
            estimates = estimates.mean(dim=2, keepdim=True)
            candidates = candidates.mean(dim=2, keepdim=True)
        if self.center:
            estimates = estimates - estimates.mean(dim=(1, 2), keepdim=True)
            candidates = candidates - candidates.mean(dim=(1, 2), keepdim=True)
        inv_norms = 1 / (1e-8 + candidates.norm(dim=(1, 2), p=2))
        # We normalize inside the einsum, to avoid creating a copy
        # of candidates, which can be pretty big.
        scores = torch.einsum("bct,oct,o->bo", estimates, candidates, inv_norms)
        return scores

    def get_probabilities(self, estimates, candidates):
        """Given estimates that is [B, C, T] and candidates
        which is [B', C, T], return a [B, B'] matrix of probabilities of matching.
        """
        scores = self.get_scores(estimates, candidates)
        return F.softmax(scores, dim=1)

    def forward(self, estimate, candidate, mask=None):
        """Warning: estimate and candidate are not symmetrical.
        If estimate of shape [B, C, T] and candidate of size [B', C, T]
        with B'>=B, the first B samples of candidate are targets, while
        the remaining B'-B samples of candidate are only used as negatives.
        """
        # assert mask.all(), "mask is not supported for now"
        assert estimate.size(0) <= candidate.size(0), "need at least as many targets as estimates"
        scores = self.get_scores(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)
        return F.cross_entropy(scores, target)


class ConvSequence(nn.Module):

    def __init__(self, channels: tp.Sequence[int], kernel: int = 4, dilation_growth: int = 1,
                 dilation_period: tp.Optional[int] = None, stride: int = 2,
                 dropout: float = 0.0, leakiness: float = 0.0, groups: int = 1,
                 decode: bool = False, batch_norm: bool = False, dropout_input: float = 0,
                 skip: bool = False, scale: tp.Optional[float] = None, rewrite: bool = False,
                 activation_on_last: bool = True, post_skip: bool = False, glu: int = 0,
                 glu_context: int = 0, glu_glu: bool = True, activation: tp.Any = None) -> None:
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(Conv(chin, chout, kernel, stride, pad,
                               dilation=dilation, groups=groups if k > 0 else 1))
            dilation *= dilation_growth # dilation_growth = 2
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context), act))
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


def pad_multiple(x: torch.Tensor, base: int):
    length = x.shape[-1]
    target = math.ceil(length / base) * base
    return torch.nn.functional.pad(x, (0, target - length))


class ScaledEmbedding(nn.Module):
    """Scale up learning rate for the embedding, otherwise, it can move too slowly.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        return self.embedding(x) * self.scale


class DualPathRNN(nn.Module):
    def __init__(self, channels: int, depth: int, inner_length: int = 10):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(channels, channels, 1) for _ in range(depth * 4)])
        self.inner_length = inner_length

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        IL = self.inner_length
        x = pad_multiple(x, self.inner_length)
        x = x.permute(2, 0, 1).contiguous()
        for idx, lstm in enumerate(self.lstms):
            y = x.reshape(-1, IL, B, C)
            if idx % 2 == 0:
                y = y.transpose(0, 1).reshape(IL, -1, C)
            else:
                y = y.reshape(-1, IL * B, C)
            y, _ = lstm(x)
            if idx % 2 == 0:
                y = y.reshape(IL, -1, B, C).transpose(0, 1).reshape(-1, B, C)
            else:
                y = y.reshape(-1, B, C)
            x = x + y

            if idx % 2 == 1:
                x = x.flip(dims=(0,))
        return x[:L].permute(1, 2, 0).contiguous()

class PositionGetter:
    INVALID = -0.1

    def __init__(self) -> None:
        self._cache: tp.Dict[int, torch.Tensor] = {}
        self._invalid_names: tp.Set[str] = set()

    def get_recording_layout(self, layout) -> torch.Tensor:
        # index = recording.recording_index
        # if index in self._cache:
        #    return self._cache[index]
        # else:
            # info = recording.mne_info
            # layout = mne.find_layout(info)
        indexes: tp.List[int] = []
        valid_indexes: tp.List[int] = []
        for meg_index, name in enumerate(layout.names):
            name = name.rsplit("-", 1)[0]
            try:
                indexes.append(layout.names.index(name))
            except ValueError:
                if name not in self._invalid_names:
                    print(
                        "Channels %s not in layout for recording",
                        name,)
                    self._invalid_names.add(name)
            else:
                valid_indexes.append(meg_index)

        positions = torch.full((len(layout.names), 2), self.INVALID)
        x, y = layout.pos[indexes, :2].T
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        positions[valid_indexes, 0] = x
        positions[valid_indexes, 1] = y
        # positions = positions.repeat(64, 1, 1) # 64 is batch
        # self._cache[index] = positions
        return positions

    def get_positions(self, meg, batch, layout):
        # meg = batch.meg
        B, C, T = meg.shape
        positions = torch.full((B, C, 2), self.INVALID, device=meg.device)
        for idx in range(len(batch['subject_index'])):
            # recording = batch._recordings[idx]
            rec_pos = self.get_recording_layout(layout)
            positions[idx, :len(rec_pos)] = rec_pos.to(meg.device)
        return positions

    def is_invalid(self, positions):
        return (positions == self.INVALID).all(dim=-1)

class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sinuses, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """

    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2) ** 0.5
        assert int(n_freqs ** 2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2) ** 0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat([
            torch.cos(loc),
            torch.sin(loc),
        ], dim=-1)
        return emb

class ChannelMerger(nn.Module):
    def __init__(self, chout: int, pos_dim: int = 256,
                 dropout: float = 0, usage_penalty: float = 0.,
                 n_subjects: int = 200, per_subject: bool = False):
        super().__init__()
        assert pos_dim % 4 == 0
        self.position_getter = PositionGetter()
        self.per_subject = per_subject
        if self.per_subject:
            self.heads = nn.Parameter(torch.randn(n_subjects, chout, pos_dim, requires_grad=True))
        else:
            self.heads = nn.Parameter(torch.randn(chout, pos_dim, requires_grad=True))
        self.heads.data /= pos_dim ** 0.5
        self.dropout = dropout
        self.embedding = FourierEmb(pos_dim)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def forward(self, meg, batch, layout):
        B, C, T = meg.shape
        meg = meg.clone()
        positions = self.position_getter.get_positions(meg, batch, layout)
        embedding = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=meg.device)
        # score_offset[self.position_getter.is_invalid(positions)] = float('-inf')

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=meg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float('-inf')
        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            subject = batch['subject_index']-1 # -1?
            heads = self.heads.gather(0, subject.view(-1, 1, 1).expand(-1, cout, pos_dim))
        else:
            heads = self.heads[None].expand(B, -1, -1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", meg, weights)
        if self.training and self.usage_penalty > 0.:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out

class SubjectLayers(nn.Module):
    """Per subject linear layer."""

    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.C = in_channels
        self.D = out_channels
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels), requires_grad=True)
        #self.weights = nn.Embedding(n_subjects, in_channels*out_channels)
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        # self.weights.data *= 1 / in_channels ** 0.5
    def forward(self, x, subjects):
        B = x.shape[0]
        weights = self.weights.gather(0, subjects.long().view(-1, 1, 1).expand(-1, self.C, self.D))
        #weights = self.weights(subjects).view(B, self.C, self.D)
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"

import pickle
with open('layout.pkl', 'rb') as f:
    loaded_layout = pickle.load(f)
# loaded_layout.pos = torch.from_numpy(loaded_layout.pos).repeat(64, 1, 1) # 64 is batch
class SimpleConv(nn.Module):
    def __init__(self,
                 # Channels
                 in_channels: tp.Dict[str, int] = {"meg": 208},
                 out_channels: int = 80,
                 hidden: tp.Dict[str, int] = {"meg": 320},
                 # Overall structure
                 depth: int = 5,
                 concatenate: bool = False,  # concatenate the inputs
                 linear_out: bool = False,
                 complex_out: bool = True,
                 # Conv layer
                 kernel_size: int = 3,
                 growth: float = 1.,
                 dilation_growth: int = 2,
                 dilation_period: tp.Optional[int] = 5,
                 skip = True, 
                 post_skip: bool = False,
                 scale: tp.Optional[float] = None,
                 rewrite: bool = False,
                 groups: int = 1,
                 glu: int = 2,
                 glu_context: int = 1,
                 glu_glu: bool = True,
                 gelu: bool = True,
                 # Dual path RNN
                 dual_path: int = 0,
                 # Dropouts, BN, activations
                 conv_dropout: float = 0.0,
                 dropout_input: float = 0.0,
                 batch_norm: bool = True,
                 relu_leakiness: float = 0.0,
                 # Subject specific settings
                 n_subjects: int = 27,
                 subject_dim: int = 0,
                 subject_layers: bool = True,
                 subject_layers_dim: str = "hidden",  # or hidden
                 subject_layers_id: bool = False,
                 embedding_scale: float = 1.0,
                 # stft transform
                 n_fft: tp.Optional[int] = None,
                 fft_complex: bool = True,
                 # Attention multi-dataset support
                 merger: bool = True,
                 merger_pos_dim: int = 2048,
                 merger_channels: int = 270,
                 merger_dropout: float = 0.2,
                 merger_penalty: float = 0.,
                 merger_per_subject: bool = False,
                 dropout: float = 0.,
                 dropout_rescale: bool = True,
                 initial_linear: int = 270,
                 initial_depth: int = 1,
                 initial_nonlin: bool = False,
                 subsample_meg_channels: int = 0,
                 run_name = 'default'
                 ):
        super().__init__()
        self.run_name= run_name
        if set(in_channels.keys()) != set(hidden.keys()):
            raise ValueError("Channels and hidden keys must match "
                             f"({set(in_channels.keys())} and {set(hidden.keys())})")
        self._concatenate = concatenate
        self.out_channels = out_channels
        if gelu:
            activation = nn.GELU
        elif relu_leakiness:
            activation = partial(nn.LeakyReLU, relu_leakiness)
        else:
            activation = nn.ReLU

        assert kernel_size % 2 == 1, "For padding to work, this must be verified"

        self.subsampled_meg_channels: tp.Optional[list] = None
        if subsample_meg_channels:
            assert 'meg' in in_channels
            indexes = list(range(in_channels['meg']))
            rng = random.Random(1234)
            rng.shuffle(indexes)
            self.subsampled_meg_channels = indexes[:subsample_meg_channels]
        
        self.layout=loaded_layout # only gwilliams
        self.merger = None
        if merger:
            self.merger = ChannelMerger(
                merger_channels, pos_dim=merger_pos_dim, dropout=merger_dropout,
                usage_penalty=merger_penalty, n_subjects=n_subjects, per_subject=merger_per_subject)
            in_channels["meg"] = merger_channels

        self.initial_linear = None
        if initial_linear:
            init = [nn.Conv1d(in_channels["meg"], initial_linear, 1)]
            for _ in range(initial_depth - 1):
                init += [activation(), nn.Conv1d(initial_linear, initial_linear, 1)]
            if initial_nonlin:
                init += [activation()]
            self.initial_linear = nn.Sequential(*init)
            in_channels["meg"] = initial_linear

        self.subject_layers = None
        if subject_layers:
            assert "meg" in in_channels
            meg_dim = in_channels["meg"]
            dim = {"hidden": hidden["meg"], "input": meg_dim}[subject_layers_dim]
            self.subject_layers = SubjectLayers(meg_dim, dim, n_subjects, subject_layers_id)
            in_channels["meg"] = dim

        self.stft = None
        if n_fft is not None:
            assert "meg" in in_channels
            self.fft_complex = fft_complex
            self.n_fft = n_fft
            self.stft = ta.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=n_fft // 2,
                normalized=True,
                power=None if fft_complex else 1,
                return_complex=True)
            in_channels["meg"] *= n_fft // 2 + 1
            if fft_complex:
                in_channels["meg"] *= 2

        self.subject_embedding = None
        if subject_dim:
            self.subject_embedding = ScaledEmbedding(n_subjects, subject_dim, embedding_scale)
            in_channels["meg"] += subject_dim

        # concatenate inputs if need be
        if concatenate:
            in_channels = {"concat": sum(in_channels.values())}
            hidden = {"concat": sum(hidden.values())}

        # compute the sequences of channel sizes
        sizes = {}
        for name in in_channels:
            sizes[name] = [in_channels[name]]
            sizes[name] += [int(round(hidden[name] * growth ** k)) for k in range(depth)]
        params: tp.Dict[str, tp.Any]
        params = dict(kernel=kernel_size, stride=1,
                      leakiness=relu_leakiness, dropout=conv_dropout, dropout_input=dropout_input,
                      batch_norm=batch_norm, dilation_growth=dilation_growth, groups=groups,
                      dilation_period=dilation_period, skip=skip, post_skip=post_skip, scale=scale,
                      rewrite=rewrite, glu=glu, glu_context=glu_context, glu_glu=glu_glu,
                      activation=activation)

        final_channels = sum([x[-1] for x in sizes.values()])
        self.dual_path = None
        if dual_path:
            self.dual_path = DualPathRNN(final_channels, dual_path)
        self.final = None
        pad = 0
        kernel = 1
        stride = 1
        if n_fft is not None:
            pad = n_fft // 4
            kernel = n_fft
            stride = n_fft // 2

        #self.final = nn.Conv1d(final_channels, out_channels, kernel_size=3, stride=2, )
        #self.fc=nn.Linear(80,80)
        #self.fc1=nn.Linear(80,80)
        if linear_out:
            assert not complex_out
            self.final = nn.ConvTranspose1d(final_channels, out_channels, kernel, stride, pad)
        elif complex_out:
            self.final = nn.Sequential(
                nn.Conv1d(final_channels, 2 * final_channels, 1),
                activation(),
                nn.ConvTranspose1d(2 * final_channels, out_channels, kernel, stride, pad))
        else:
            assert len(sizes) == 1, "if no linear_out, there must be a single branch."
            params['activation_on_last'] = False
            list(sizes.values())[0][-1] = out_channels

        self.encoders = nn.ModuleDict({name: ConvSequence(channels, **params)
                                       for name, channels in sizes.items()})

    def forward(self, inputs, batch):
        # subj idx,
        subjects = batch['subject_index']-1
        length = inputs["meg"].shape[2]  # length of any of the inputs

        if self.subsampled_meg_channels is not None:
            mask = torch.zeros_like(inputs["meg"][:1, :, :1])
            mask[:, self.subsampled_meg_channels] = 1.
            inputs["meg"] = inputs["meg"] * mask
        
        if self.merger is not None:
            inputs["meg"] = self.merger(inputs["meg"], batch, self.layout)
        
        if self.initial_linear is not None:
            inputs["meg"] = self.initial_linear(inputs["meg"])

        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)

        if self.stft is not None:
            x = inputs["meg"]
            pad = self.n_fft // 4
            x = F.pad(pad_multiple(x, self.n_fft // 2), (pad, pad), mode='reflect')
            z = self.stft(inputs["meg"])
            B, C, Fr, T = z.shape
            if self.fft_complex:
                z = torch.view_as_real(z).permute(0, 1, 2, 4, 3)
            z = z.reshape(B, -1, T)
            inputs["meg"] = z

        if self.subject_embedding is not None:
            emb = self.subject_embedding(subjects)[:, :, None]
            inputs["meg"] = torch.cat([inputs["meg"], emb.expand(-1, -1, length)], dim=1)

        if self._concatenate:
            input_list = [x[1] for x in sorted(inputs.items())]
            inputs = {"concat": torch.cat(input_list, dim=1)}

        encoded = {}
        for name, x in inputs.items():
            encoded[name] = self.encoders[name](x)
        inputs = [x[1] for x in sorted(encoded.items())]
        x = torch.cat(inputs, dim=1)
        if self.final is not None:
            x = self.final(x)
        #x=F.gelu(x)
        #bs,ch,l=x.shape
        #x=torch.permute(x,(0,2,1))
        #x = self.fc(x.reshape([bs*l,ch]))
        #x = F.gelu(x)
        #x = self.fc1(x)
        #x = x.reshape([bs,l,ch])
        #x=torch.permute(x,(0,2,1))
        #assert x.shape[-1] >= length
        return x[:, :, :length]


if __name__ == '__main__':
    model = SimpleConv()
    # Input and output are of the same length
    # EEG is 100 points per second
    # mel is 100 points in 1s
    # Define input data
    # It is assumed here that the input data is a dictionary containing the input tensor of the 'meg' channel, with a length of 10
    inputs = {'meg': torch.randn(1, 270, 1000)}

    # Define batch data, only one 'subject_index' is needed here as an example
    batch = {'subject_index': torch.tensor([0])}

    # perform forward propagation
    outputs = model(inputs, batch)

    # The input and output shapes of the output model
    print("Enter shape:", inputs['meg'].shape)
    print("Output shape:", outputs.shape)
    print(model)
