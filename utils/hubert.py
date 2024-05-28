# hubert tokenizer -- convert wav to token ids

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import json
import joblib
from fairseq import utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
# from examples.hubert.simple_kmeans.dump_hubert_feature import HubertFeatureReader
import soundfile as sf
import torch.nn.functional as F
from fairseq.examples.hubert.simple_kmeans.dump_km_label import ApplyKmeans
from fairseq.checkpoint_utils import load_model_ensemble_and_task
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
class HubertFeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer, max_chunk=1600000,batch_size=64):
        (
            model,
            cfg,
            task,
        ) = load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.batch_size = batch_size

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        assert sr == self.task.cfg.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path, ref_len=None):
        x = self.read_audio(file_path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)

    def get_feats_from_paths(self,file_paths):
        x=[torch.from_numpy(self.read_audio(path)).float() for path in file_paths]
        x=torch.stack(x,dim=0).cuda()
        assert x.shape[0]<=self.batch_size
        with torch.no_grad():
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            assert len(x.shape)==2,f'x shape:{x.shape}'
            x, _ = self.model.extract_features(
                source=x,
                padding_mask=None,
                mask=False,
                output_layer=self.layer,
            )
        return x  # [bs,featlen]


# Hubert tokenizer
class HubertTokenizer:
    def __init__(
                self,
                hubert_path,
                hubert_layer,
                km_path,batch_size
            ):
        self.feature_extractor = HubertFeatureReader(hubert_path, hubert_layer,batch_size=batch_size)
        self.quantizer = joblib.load(open(km_path, "rb"))
        self.quantizer.verbose = False

    def wav2code(self, path):
        feat = self.feature_extractor.get_feats(path)
        code = self.quantizer.predict(feat.cpu().numpy())
        return code.tolist()

    def wavs2code(self, paths):
        feats = self.feature_extractor.get_feats_from_paths(paths).detach().cpu().numpy()
        codes = [self.quantizer.predict(feat) for feat in feats]
        return codes



class CnnRnnClassifier(torch.nn.Module):
    """
    The CNN RNN classifier that I used for the bravo1 decoding, in pytorch.
    """

    def __init__(self, rnn_dim, KS, num_layers, dropout, n_classes, bidirectional, in_channels, keeptime=False,
                 token_input=None):
        super().__init__()

        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                            out_channels=rnn_dim,
                                            kernel_size=KS,
                                            stride=KS)

        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.keeptime = keeptime

        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else:
            mult = 1
        self.mult = mult
        if keeptime:
            self.postprocessing_conv = nn.ConvTranspose1d(in_channels=rnn_dim * mult,
                                                          out_channels=rnn_dim * mult,
                                                          kernel_size=KS,
                                                          stride=KS)
        self.dense = nn.Linear(rnn_dim * mult, n_classes)

    def forward(self, x):
        # x comes in bs, t, c
        x = x.contiguous().permute(0, 2, 1)
        # now bs, c, t
        x = self.preprocessing_conv(x)
        #         x = F.relu(x)
        x = self.dropout(x)
        x = x.contiguous().permute(2, 0, 1)
        # now t, bs, c
        output, x = self.BiGRU(x)  # output: t,bs,d*c
        if not self.keeptime:
            x = x.contiguous().view(self.num_layers, self.mult, -1, self.rnn_dim)
            # (2, bs, rnn_dim)
            x = x[-1]  # Only care about the output at the final layer.
            # (2, bs, rnn_dim)
            x = x.contiguous().permute(1, 0, 2)
            x = x.contiguous().view(x.shape[0], -1)
        else:
            x = output.contiguous().permute(1, 2, 0)  # bs,d*c,t
            x = self.postprocessing_conv(x)
            x = x.permute(0, 2, 1)  # bs,t,d*c

        x = self.dropout(x)
        out = self.dense(x)
        return out


class Bravo1(PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.model = CnnRnnClassifier(**config.model)
