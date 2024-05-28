import json

import mne
import numpy as np
import mne.io
import soundfile as sf
import pandas as pd
from sklearn.preprocessing import RobustScaler
import jsonlines
import os
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer)
import torch
import librosa
from IPython.display import Audio, display
import tqdm

# 主要功能描述
# 这个主要是为了创建相同长度的语音，以及不同情况的截取情况，这样能增加泛化性。

# 主要控制参数有
# 语音截取长度

# 由于每个task对应一个meg和一些分段的语音，因此，可以按照划分后的语音进行切割

# 划分后的可以再去split，这里不管
# 主要的工作流是
# 1，找到所有的tsv文件
# 2，在每个tsv文件中找到播放的几个音频文件及其对应的转录文件
# 3，将音频文件和脑电对应进行切割，滑动1s±0.5s进行切割，每个切4s的片段，并且以这个时间找到对应的words进行标注。
# 4，先将切割后的片段进行保存，保存完了之后再去识别。
# 5，将每一段语音单独拿去做语音识别。形成最后的jsonl 文件

# 升级方案，就不切割数据，到dataloader里面去做。

from torch.nn import functional as F
import torchaudio
import numpy as np
import math
import warnings
import julius

def extract_words_in_range(words, start_time, end_time):
    """
    从给定的时间范围内提取单词。

    参数:
    words (list): 单词列表，每个单词是一个字典，包含'word', 'start', 'end'键。
    start_time (float): 时间范围的开始时间。
    end_time (float): 时间范围的结束时间。

    返回:
    list: 在时间范围内的单词列表。
    """
    # 筛选出在指定时间范围内的单词
    # 这个有点坑，如果是数字之类的，他不给start 和 end 的键
    start_index = None
    end_index = None

    for i, word in enumerate(words):
        if 'start' in word and word['start'] >= start_time:
            if start_index is None:
                start_index = i

        if 'end' in word and word['end'] >= end_time:
            if end_index is None:
                end_index = i

    if start_index is not None and end_index is not None:
        return words[start_index:end_index + 1]
    elif start_index is not None:
        return words[start_index:]
    elif end_index is not None:
        return words[:end_index + 1]
    else:
        return []


def get_sequences(tsv_path):
    text = pd.read_csv(tsv_path, delimiter='\t')
    slice_dict_list = []
    for i in range(len(text)):
        tti = eval(text['trial_type'][i])
        if 'sequence_id' not in tti.keys():
            tti['sound'] = os.path.basename(tti['sound']).split('.')[0].lower()
            tti['story'] = tti['story'].lower()

            tti1 = eval(text['trial_type'][i + 1])
            tti['speech_rate'] = tti1['speech_rate']
            tti['voice'] = tti1['voice']

            slice_dict = {'sample': text.iloc[i]['sample'],
                          'onset': text.iloc[i]['onset'],
                          **tti}
            slice_dict_list.append(slice_dict)

    sentences = []

    # 合并几个 slice 的转录文本
    # 先找到所有转录文本
    # 过滤出故事对饮的转录文本
    # 按照顺序对文本排列
    # 合并文本
    # 给出每个句子的信息
    audio_folder_path = f'/data/johj/MEG/gwilliams2023/preprocess8/audio'
    wav_dir = os.path.join(audio_folder_path, 'wav16')
    transcription_dir = os.path.join(audio_folder_path, 'transcription')
    # transcription_files=os.listdir(transcription_dir)
    # story=slice_dict_list[0]['story'].lower()
    # transcription_files=[file for file in transcription_files if file.startswith(story)]
    # transcription_files=sorted(transcription_files, key=lambda x: int(x[len(story):].split('_')[0]))

    # 从slice_dict中去读取顺序播放的音频文件
    # 找到对应的文本文件
    # 将信息按句子级别抽取保存
    for slice_dict in slice_dict_list:
        transcription_file_name = slice_dict['sound'] + '_16kHz.json'
        transcription_file_name = os.path.join(transcription_dir, transcription_file_name)
        audio_file_name = slice_dict['sound'] + '_16kHz.wav'
        audio_file_name = os.path.join(wav_dir, audio_file_name)

        transcription = read_json(transcription_file_name)
        language = transcription['language']
        onset = slice_dict['onset']  # the indices of eeg
        # print(f'transcription:{len(segments)}')
        # 重写这一段代码即可，在这里滑动取词。
        # 词语根据这里的转录文本的单词时间戳产生。因此不需要按照这里的segments进行循环。
        # 我们需要先将转录本中的单词全部提取并转换为一个列表。  可能需要重新使用whisper-x进行标注，据说单词的时间戳会准确一些。

        words = [word for seg in transcription['segments'] for word in seg['words']]
        # 补全字典
        # for i,word in enumerate(words):
        #     if 'start' not in word:
        #         word['start']=words[i-1]['end']
        #         word['end']=words[i+1]['start']
        assert len(words)>0
        wav, sr = sf.read(audio_file_name, always_2d=True)

        audio_secs = wav.shape[0] / sr
        slide_sec = 1
        seg_sec = 3 # modify 05014

        for start_sec in range(0, int(np.ceil(audio_secs - seg_sec)), slide_sec):
            sent = {}
            start_sec += np.random.uniform(high=1) - 0.5
            start_sec = np.clip(start_sec, 0, audio_secs - seg_sec)
            end_sec = start_sec + seg_sec
            # words 需要根据时间选出来
            # print(words)
            try:
                selected_words = extract_words_in_range(words, start_time=start_sec, end_time=end_sec)
            except Exception as e:

                print(f'{transcription_file_name}出错了')
                print(e)
                print(words[0].keys())
                # selected_words=[word for word in words if word['start'] >= start_sec and word['end'] <= end_sec]
                # print(selected_words)
                for i,word in enumerate(words):
                    print(i,word)
                    print('start',word['start'])
                    # if word['start'] >= start_sec and word['end'] <= end_sec:
                    #     print(word)
                print(slice_dict)
                print(sent)
                raise Exception

            sent['words'] = selected_words
            sent['text'] = ' '.join([word['word'] for word in selected_words])
            sent['language'] = language
            sent['audio_start'] = start_sec
            sent['audio_end'] = end_sec
            sent['start'] = start_sec + onset
            sent['end'] = end_sec + onset
            sent['duration'] = seg_sec

            sent['story'] = slice_dict['story']
            sent['story_id'] = slice_dict['story_uid']
            sent['sound_id'] = slice_dict['sound_id']
            sent['speech_rate'] = slice_dict['speech_rate']
            sent['voice'] = slice_dict['voice']

            sent['meg_path'] = tsv_path[:-10] + 'meg.con'
            sent['audio_path'] = audio_file_name
            sentences.append(sent)

    return sentences


def preprocess_eeg_data(data, threshold=10):
    data = data.T

    # 2. Robust scaling
    scaler = RobustScaler()
    scaler.fit(data[:60]) # change 500 => 60
    data = scaler.transform(data).T
    # 3. Clipping outliers

    data[np.abs(data) > threshold] = np.sign(data[np.abs(data) > threshold]) * threshold
    data = data / threshold
    threshold_mask = np.abs(data) > 1
    num_clipped = np.sum(threshold_mask)

    # 计算比例
    clipped_ratio = num_clipped / (data.shape[0] * data.shape[1])
    assert clipped_ratio < 0.2, 'clip ratio should below 20%'
    return data, clipped_ratio


def find_files_with_extension(folder_path, extension):
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(file_path)

    return file_paths


def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)


def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path


# 按照句子的时间去切割
def process_meg(tsv_path):
    print(tsv_path,'begin')
    target_meg_sr = 120 # change 200 => 120
    sentences = get_sequences(tsv_path)
    save_sentences_path=tsv_path.replace('.tsv','save_sentences_info.jsonl')
    assert save_sentences_path!=tsv_path,' these two have to be different'
    write_jsonlines(save_sentences_path,sentences)
    # print(tsv_path,len(sentences))
    meg_path = sentences[0]['meg_path']
    meg = mne.io.read_raw_kit(meg_path, preload=True, verbose=False)
    picks = mne.pick_types(
        meg.info, meg=True, ref_meg=False, eeg=False, stim=False, eog=False, ecg=False
    )
    meg.pick(picks, verbose=False)
    # meg.notch_filter(60, verbose=False)
    meg.filter(l_freq=1, h_freq=58, verbose=False)
    meg.resample(target_meg_sr)
    data = meg.get_data()
    assert data.shape[0] == 208, f'data shape:{data.shape}'
    old_audio_path = None
    lines = []

    for i, sent in enumerate(sentences):
        # 切分meg
        start_meg_index = int(sent['start'] * target_meg_sr)
        end_meg_index = int(sent['end'] * target_meg_sr)
        seg_meg = data[:, start_meg_index:end_meg_index]
        sent['duration'] = seg_meg.shape[1] / target_meg_sr
        # 切分音频
        audio_path = sent['audio_path']
        if old_audio_path != audio_path:
            speech_data, speech_sr = sf.read(os.path.join(folder_path, audio_path))
        start_audio_index = int(sent['audio_start'] * speech_sr)
        end_audio_index = int(sent['audio_end'] * speech_sr)
        seg_audio = speech_data[start_audio_index:end_audio_index]
        #
        
        mel = processor(audio=seg_audio, sampling_rate=16000,
                             return_tensors="pt", return_attention_mask=True)
        speech_mel_input_features = mel.input_features
        speech_mel_useful_length = torch.sum(mel.attention_mask).item()
        

        # 标准化
        seg_meg, cr = preprocess_eeg_data(seg_meg, threshold=20) # change threshold=20 for the same setting as meta
        # 将处理好的音频文件，脑电文件，标注文件都储存好。
        seg_meg_path = tsv_path.replace('download', replace_folder).replace('events.tsv', f'senid_{i}_meg.npy')
        seg_audio_path = seg_meg_path.replace('meg.npy', 'audio.wav')
        seg_mel_path = seg_meg_path.replace('meg.npy', 'mel.npy')
        makedirs(seg_meg_path)
        # print(f'{i} seg_meg {seg_meg.shape} seg_meg_path {seg_meg_path}')
        np.save(seg_meg_path, seg_meg)
        np.save(seg_mel_path, speech_mel_input_features)
        # seg_meg = np.load(seg_meg_path)
        sf.write(seg_audio_path, seg_audio, target_speech_sr)

        # 解析其他的键值对
        selected_keys = ['story', 'story_id', 'sound_id', 'speech_rate', 'voice']

        new_dict = {key: sent[key] for key in selected_keys}
        # 做 whisper json
        line = {
            "speech": {"path": seg_audio_path, 'sr': target_speech_sr},
            "eeg": {"path": seg_meg_path, 'sr': target_meg_sr},
            "mel": {"path": seg_mel_path, 'speech_mel_useful_length': speech_mel_useful_length},
            "audio_start": 0,
            "audio_end": sent['audio_end'] - sent['audio_start'],
            "start": 0,
            "end": sent['duration'],
            "duration": sent['duration'],
            # "delay_sec": delay_sec,
            "language": "English",
            "sentence": sent['text'],
            "sentences": [{"text": sent['text'],
                           "start": 0.0, "end": sent['duration'], "duration": sent['duration'],
                           "words": [{"word": word['word'],
                                      "start": word['start'] - sent['audio_start'] if 'start' in word.keys() else None,
                                      "end": word['end'] - sent['audio_start'] if 'end' in word.keys() else None}
                                     for word in sent['words']]}],
            "subj": int(os.path.basename(tsv_path)[4:6]),
            **new_dict
        }
        lines.append(line)
    if len(lines) != 0:
        seg_jsonl_path = tsv_path.replace('download', replace_folder).replace('events.tsv', 'info.jsonl')
        write_jsonlines(seg_jsonl_path, lines)
    print(tsv_path,'done')
    return lines


from multiprocessing import Pool


def read_json(file_path):
    with open(file_path, 'r') as f:
        file = json.load(f)
    return file


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts


def process_file(filename_id):
    lines = process_meg(events_tsv_list[filename_id])
    return lines


# python process_dataset/gwilliams2023_process_240411.py
if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    replace_folder = 'preprocess8'
    folder_path = '/data/johj/MEG/gwilliams2023/'
    audio_folder_path = f'/data/johj/MEG/gwilliams2023/{replace_folder}/audio'
    base_model = 'openai/whisper-base'
    language = 'en'
    task = 'transcribe'
    timestamps = False
    local_files_only = True
    extension = 'events.tsv'
    original_eeg_sr = 1000
    target_speech_sr = 16000
    delay_sec = 0.5
    events_tsv_list = find_files_with_extension(folder_path, extension)
    processor = WhisperProcessor.from_pretrained(base_model,
                                                 language=language,
                                                 task=task,
                                                 no_timestamps=not timestamps,
                                                 local_files_only=local_files_only, hop_length=128) # equal to Meta 
    # results=[process_file(file) for file in events_tsv_list[:2]]

    pool = Pool(processes=16)
    results = pool.map(process_file, np.arange(len(events_tsv_list)))
    pool.close()
    pool.join()

    all_lines = []
    for lines in results:
        all_lines.extend(lines)

    write_jsonlines(os.path.join(folder_path.replace('download', replace_folder), 'info.jsonl'), all_lines)

    # 将数据分为训练集，验证集和测试集。
    # 第一种分法是将所有数据直接随机分为8:1:1
    # 第二种分法是将session1的数据分为训练和验证9:1，测试集为整个session2的数据。
    # import random
    #
    # data = read_jsonlines(f'/hpc2hdd/home/yyang937/datasets/gwilliams2023/{replace_folder}/info.jsonl')  # 替换为你的数据列表
    # random.shuffle(data)  # 随机打乱数据列表
    #
    # total_samples = len(data)
    # train_samples = int(0.8 * total_samples)
    # val_samples = int(0.1 * total_samples)
    # test_samples = total_samples - train_samples - val_samples
    #
    # train_data1 = data[:train_samples]
    # val_data1 = data[train_samples:train_samples + val_samples]
    # test_data1 = data[train_samples + val_samples:]
    #
    # print("训练集大小:", len(train_data1))
    # print("验证集大小:", len(val_data1))
    # print("测试集大小:", len(test_data1))
    # split1_path = os.path.join(folder_path.replace('download', replace_folder), 'split1')
    # os.makedirs(split1_path, exist_ok=True)
    # write_jsonlines(os.path.join(split1_path, 'train.jsonl'), train_data1)
    # write_jsonlines(os.path.join(split1_path, 'val.jsonl'), val_data1)
    # write_jsonlines(os.path.join(split1_path, 'test.jsonl'), test_data1)
