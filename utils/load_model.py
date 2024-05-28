# coding=utf-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Whisper model."""

from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch.nn import MSELoss
import torch.nn.functional as F
import re
from transformers.models.whisper.modeling_whisper import *
from transformers.utils import ModelOutput, logging
from utils.brain_module import SimpleConv, ClipLoss
from utils.loss import MMDLoss
from dataclasses import dataclass
from transformers import PreTrainedModel, WhisperProcessor
from typing import Optional, Tuple, Union
logger = logging.get_logger(__name__)

def print_tensor_features(tensor1, tensor2):
    # Compute the numerical characteristics of tensor 1
    max_value1 = tensor1.max()
    min_value1 = tensor1.min()
    mean_value1 = tensor1.mean()
    std_value1 = tensor1.std()
    range_value1 = max_value1 - min_value1
    size1 = tensor1.numel()  # Data size

    # Compute the numerical characteristics of tensor 2
    max_value2 = tensor2.max()
    min_value2 = tensor2.min()
    mean_value2 = tensor2.mean()
    std_value2 = tensor2.std()
    range_value2 = max_value2 - min_value2
    size2 = tensor2.numel()  # Data size

    # Calculate the gap and associated features between two tensors
    max_difference = max(max_value1, max_value2) - min(min_value1, min_value2)
    mean_difference = mean_value1 - mean_value2
    std_difference = std_value1 - std_value2

    # Print numerical features of two tensors
    print("Tensor 1:")
    print(f"Max: {max_value1}, Min: {min_value1}, Mean: {mean_value1}, Std: {std_value1}, Range: {range_value1}, Size: {size1}")
    print("Tensor 2:")
    print(f"Max: {max_value2}, Min: {min_value2}, Mean: {mean_value2}, Std: {std_value2}, Range: {range_value2}, Size: {size2}")

    # Print the gap and associated features between two tensors
    print("Differences:")
    print(f"Max Difference: {max_difference}, Mean Difference: {mean_difference}, Std Difference: {std_difference}")

def match_modules_string(named_modules,
                         start_prefixes,
                         end_suffixes,
                         mid_prefixes=[], ):
    matched_modules = []

    for name, _ in named_modules:

        start_matched = False
        for start in start_prefixes:
            if name.startswith(start):
                start_matched = True
                break

        if not start_matched:
            continue

        if mid_prefixes:
            mid_matched = False
            for mid in mid_prefixes:
                if mid in name:
                    mid_matched = True
                    break

            if not mid_matched:
                continue

        end_matched = False
        for end in end_suffixes:
            if name.endswith(end):
                matched_modules.append(name)
                end_matched = True
                break

        if not end_matched:
            continue

    return matched_modules

def match_modules(named_modules, prefix_list, suffix_list, mid_fix_list=['']):
    matched_modules = []
    for name, _ in named_modules:
        for prefix in prefix_list:
            for suffix in suffix_list:
                for mid_fix in mid_fix_list:
                    pattern = re.compile(fr'^({prefix}).*({mid_fix}).*({suffix})$')
                    if re.match(pattern, name):
                        matched_modules.append(name)
                        break
    return matched_modules

@dataclass
class Seq2SeqBrainLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        clip_loss: clip loss.
        p_mel: p_mel.
        mel: mel.
        subject_index: subject_index.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_component: Optional[torch.FloatTensor] = None
    p_mel: Optional[torch.FloatTensor] = None
    mel: Optional[torch.FloatTensor] = None
    subject_index: Optional[Tuple[torch.IntTensor]] = None

class BrainWhisperForConditionalGeneration2(PreTrainedModel): #nn.Module
    base_model_prefix = "model"
    _tied_weights_keys = ["brain_module"]
    def __init__(self, config, loss_dict, pretrained_layers, run_name, depth):
        super(BrainWhisperForConditionalGeneration2, self).__init__(config)
        self.config = config
        self.config.total_loss = loss_dict
        self.config.modal_ch = getattr(config,"modal_ch",208)
        self.config.mmd_input_type='mean'
        self.config.run_name = run_name
        self.config.depth = depth
        self.model = pretrained_layers
        self.brain_module = SimpleConv(in_channels={"meg": self.config.modal_ch}, run_name=self.config.run_name, depth=self.config.depth).to(self.model.device)
        self.processor = WhisperProcessor.from_pretrained('openai/whisper-base',
                                                    language='en',
                                                    task='transcribe',
                                                    local_files_only=False)
        # Initialize weights and apply final processing
        # self.model.post_init()

    def get_encoder(self):
        return self.model.get_encoder()
    
    def get_decoder(self):
        return self.model.get_decoder()
    
    def get_encoder_output(self,
                           input_features, attention_mask, head_mask,
                           output_attentions, output_hidden_states, return_dict=True):
        # input_features = self._mask_input_features(input_features, attention_mask = attention_mask)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_outputs = self.get_encoder()(
            input_features,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
            
        return Seq2SeqModelOutput(
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def predict_mel(self,input_features,useful_length,subject_index):
        seq_len = 3000
        pred_mel = self.brain_module({"meg": input_features[..., :useful_length]}, {"subject_index": subject_index})
        pad_length = seq_len - pred_mel.size(2)
        if pad_length > 0:
            # Use nn.functional.pad to fill the last dimension (time dimension)
            # The filled value can be 0 or other values, depending on your needs
            pred_mel = F.pad(pred_mel, (0, pad_length), 'constant', -10.0)
        # item_max_values = pred_mel.max(dim=1).values.max(dim=1).values  # [batchsize]

        # Create a tensor of the same shape as pred_mel, with the maximum value of each item being item_max_values
        max_values_batch = torch.max(pred_mel.flatten(start_dim=1, end_dim=2), dim=1).values
        # print(max_values_batch)
        # Make sure max_values_batch has shape [batch_size, 1]
        max_values_batch = max_values_batch.unsqueeze(-1).unsqueeze(-1)

        # Use torch.clamp to limit the value of pred_mel to the range [minimum value, max_values_tensor]
        pred_mel = torch.clip(pred_mel, min=max_values_batch - 2, max=max_values_batch)
        return pred_mel
    
    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            subject_index=None,
            mel_spec=None,
            useful_length=None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqBrainLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        #print(f'subject_index:{subject_index}')
        #print(f'useful_length:{useful_length}')
        useful_length = 400
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        '''
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        '''
        pred_mel=self.predict_mel(input_features=input_features, # 100*4
                                  useful_length=useful_length, # MEG sr is 100
                                  subject_index=subject_index)
        
        
        outputs = self.model(
            pred_mel,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,            
        )
        lm_logits = outputs.logits
        loss_components = dict()
        if isinstance(self.config.total_loss, list):
            loss=None
        else:
            if labels is not None:
                mmd_loss=0
                if self.config.total_loss['mmd'] != 0:
                    with torch.no_grad():
                        mel_input_encoder_outputs = self.get_encoder_output(input_features=mel_spec, attention_mask=attention_mask,
                                                                head_mask=head_mask, output_attentions=output_attentions,
                                                                output_hidden_states=output_hidden_states)
                        mel_input_encoder_outputs_last_hidden_state=mel_input_encoder_outputs.encoder_last_hidden_state
                        mel_input_encoder_outputs_last_hidden_state=mel_input_encoder_outputs_last_hidden_state.detach()

                    dimension=mel_input_encoder_outputs_last_hidden_state.shape[-1]
                    source=mel_input_encoder_outputs_last_hidden_state
                    target=outputs.encoder_last_hidden_state
                    if self.config.mmd_input_type=='mean':
                        for i in range(useful_length):
                            mmd_loss=mmd_loss+MMDLoss()(source=source[:,:,i],target=target[:,:,i])
                    elif self.config.mmd_input_type=='sample':
                        ll=10
                        indices = torch.randperm(mel_input_encoder_outputs_last_hidden_state.shape[1])
                        indices = indices[:ll]

                        source=source[:,indices,:]
                        target=target[:,indices,:]
                        source=source.reshape([-1,dimension])
                        target=target.reshape([-1,dimension])
                        mmd_loss=MMDLoss()(source=source,
                                        target=target)
                    else:
                        raise NotImplementedError
                    loss_components['mmd'] = self.config.total_loss['mmd']*mmd_loss
                
                if self.config.total_loss['ce'] != 0:
                    ce_loss = outputs.loss
                    loss_components['ce'] = self.config.total_loss['ce']*ce_loss

                if self.config.total_loss['clip'] != 0:
                    clip_loss = ClipLoss()(pred_mel[..., :useful_length], mel_spec[..., :useful_length])
                    loss_components['clip'] = self.config.total_loss['clip']*clip_loss

                if self.config.total_loss['mse'] != 0:
                    mse_loss=MSELoss()(pred_mel[..., :useful_length], mel_spec[..., :useful_length])
                    loss_components['mse'] = self.config.total_loss['mse']*mse_loss

                if self.config.total_loss['mmd_bm'] != 0:
                    mmd_bm_loss=0
                    mel_spec = mel_spec.permute(0, 2, 1)
                    pred_mel = pred_mel.permute(0, 2, 1)
                    for i in range(mel_spec.size(2)):
                        p = torch.rand(1)[0]
                        if p > 0.2:
                            continue
                        mmd_bm_loss = mmd_bm_loss + MMDLoss()(
                            source=mel_spec[:,:useful_length,i],
                            target=pred_mel[:,:useful_length,i])
                    mel_spec = mel_spec.permute(0, 2, 1)
                    pred_mel = pred_mel.permute(0, 2, 1)
                    loss_components['mmd_bm'] = self.config.total_loss['mmd_bm']*mmd_bm_loss
                    #mmd_loss = MMDLoss()(source=mel_spec[..., :useful_length].reshape([-1,useful_length]), target=pred_mel[..., :useful_length].reshape([-1,useful_length]))
                # If you only use ce and clip, this value will be between -1.1 and 0.9, which feels like noise.
                # After adding mse, this value range is still the same, indicating that the output range of the model is limited.
                # It is found that with clip loss, the most basic feature of numerical distribution cannot be learned.
                # If you replace clip with mse loss, you can learn the characteristics of the maximum and minimum average, but std is still wrong.
                # Try using mmd loss on mel.
                total_loss = sum(loss_components.values())
                loss=(
                    +total_loss
                    )
                print(
                    ", ".join(f"{key}: {float(loss_components[key])}" for key in loss_components)
                )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqBrainLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            loss_component=loss_components,
            p_mel=pred_mel[..., :useful_length],
            mel=mel_spec[..., :useful_length],
        )
    
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=False,
            return_timestamps=None,
            task=None,
            language=None,
            is_multilingual=None,
            prompt_ids: Optional[torch.Tensor] = None,
            return_token_timestamps=None,
            useful_length=None,
            subject_index=None,
            mel_spec=None,
            **kwargs,
    ):
        subject_index = subject_index.to(inputs.device)
        inputs = self.predict_mel(input_features=inputs,
                                useful_length=useful_length,
                                subject_index=subject_index
                                )
        outputs = self.model.generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            **kwargs,
        )

        if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
            num_frames = getattr(generation_config, "num_frames", None)
            outputs["token_timestamps"] = self._extract_token_timestamps(
                outputs, generation_config.alignment_heads, num_frames=num_frames
            )

        return outputs