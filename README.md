# MAD: Multi-Alignment MEG-to-Text Decoding
Above is our arxiv paper link
# Motivation
Previous studies used teacher forcing and did not compare with 
comprehensive metrics. 
For example, the eeg-to-text did not compare its results with 
input noise. After experiments, we found that input-noise has 
the same performance as input-eeg, which shows the incapability
of the model.

In this paper, we used non-teacher forcing to evaluate performance,
and compared our result with input-noise. It turns out that 
our model is capable of capturing text semantics from MEG signal.

# contribution
* propose to use non-teacher forcing, and comparing with noise-input
* provide powerful and simple data processing and augmentation framework for you to explore!
* provide important insights of checking if your model is learning, which is 
discarded in all previous works.

## thanks 
Thanks to Yiqian Yang for writing all the code in this version, 
and we are having more partners in next version to serve for the community!

please cite us if you used any code or feel inspired by our paper.
```bib
@article{xx,
  title={MAD: Multi-Alignment MEG-to-Text Decoding},
  author={xx},
  journal={xx},
  year={2024}
}
```

# code explanation
The basic logic of this code is to preprocess signal along with 
its speech labels in whisper format, and train M/EEG as speech.

## data preprocessing
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
python process_dataset/gwilliams2023_process_240310.py
```
### Computing Mel
We also provide pre-processing of MEG, speech, mel, text transcription. Pre-computing mel 
will speed up training greatly.
```bash
python process_dataset/gwilliams2023_process_240411.py
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






