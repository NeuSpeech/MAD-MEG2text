# MAD: Multi-Alignment MEG-to-Text Decoding

https://arxiv.org/pdf/2406.01512
Above is our arxiv paper link
![image](https://github.com/NeuSpeech/MAD-MEG2text/assets/151606332/16765c20-fa44-41b6-bde2-01508bcbee50)

# Motivation

Previous method seems to fail on unseen text, in this model, we used transfer learning to tackle this issue, it indicates an interesting insight that directly training the model with text alignment is almost useless, but with midiate representations, it can learn useful information. We aligned the Mel spectrogram and encoder output to gain better results.

# contribution

* MAD presents an end-to-end neural network design for the direct conversion of MEG
  signals into text in open-vocabulary, obviating the dependence on markers, teacher forcing,
  or pre-training, representing the initial implementation of translating raw MEG waves into
  text for unseen content.
* We are the first to investigate various alignments and demonstrate the benefits of aligning
  with speech modality rather than text modality in the MEG-to-text transcription task, offering
  significant insights for network improvement.
* Our extensive experimentation and thorough analysis of the proposed model showcase its
  effectiveness and highlight its superiority over existing methods in terms of translation
  accuracy, efficiency ,and reliability.

## thanks

please cite us if you used any code or feel inspired by our paper.

```bib
@article{yang2024mad,
  title={MAD: Multi-Alignment MEG-to-Text Decoding},
  author={Yang, Yiqian and Jo, Hyejeong and Duan, Yiqun and Zhang, Qiang and Zhou, Jinni and Lee, Won Hee and Xu, Renjing and Xiong, Hui},
  journal={arXiv preprint arXiv:2406.01512},
  year={2024}
}
```

# code explanation

The basic logic of this code is to preprocess signal along with
its speech labels in whisper format, and train M/EEG as speech.

### data preprocessing

for paper version, it will extract signal and MEG according to
time events in the dataset's original event.tsv
process_dataset/example.json shows an example of processed data

```bash
python process_dataset/gwilliams2023_process.py
```

But the original labels has some obvious flaws, many sentences
are incomplete, and contains errors such as 'III'

### ASR pipeline

So, we provide a pipeline to label the text according to speech,

```bash
python process_dataset/asr_pipeline.py --input_dir="stimuli_audio_directory"\
 --output_dir="asr_output_dir" 
```

and then process the datasets. You need to modify some configuration in this py first.

```bash
python process_dataset/gwilliams2023_process.py
```

### Split the data

Split the data according to the story. In this case, the sentences in the train, validation, and test sets do not overlap.

```bash
python process_dataset/filter_story_jsonl.py
```

## dataloader

utils.reader.AbstractDataset provides a parent dataloader for you to inherit,
you only need to overwrite the _get_list_data for your own class.

We build a simple yet powerful framework of handling various multimodal data, the core idea is to
pass the unit, which is a dict contains MEG raw, speech raw, speech mel, transcription and so on.
That's why this dataloader is so powerful and simple, it can even support images and video later on,
without rewriting the whole class. 1

# training and evaluation

used huggingface training code. I think it is not convenient for now,
It is hard to look into the training loop or evaluation loop to get
the gradient, to log some variables with epoch numbers. If someone is
interested in contributing in a better training and evaluation framework,
we will be very welcoming.

Besides, the model of HF needs thousands of lines of code to just modify
some modules, and it is not easy to implement generate method in the model
if additional modules is added in it.

# Correction

the bleu scores reported in the paper is calculated with nltk corpus bleu, 
we found that it is so different from the results using torchmetrics and 
we found out that we changed one line of code in the nltk package which 
encountered divide-by-zero problem and caused this situation. 
The nltk and torchmetrics scores has no difference when it is not modified.

We have to admit that this is actually not acceptable.

New results can be reproduced with `repeat_result.ipynb` with the `final_result.jsonl`.

The torchmetrics result is 6.86. the fixed nltk result is 10.49.

But the data preprocessing part is not the same with our setting, since we changed many versions, 
so the reproducing is in trouble.
We are preparing the upload the processed data to huggingface, 
we don't know if it violates the original dataset's rights,
we will delete it if we are asked to delete.


# Contact

Please do not hesitate to send me email and start collaborate with us!!!
I have more ideas now.
yyang937@connect.hkust-gz.edu.cn
