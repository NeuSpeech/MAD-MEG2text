import json
import jsonlines
import os
import sys
import soundfile as sf
import numpy as np
import copy
import mne
from collections import OrderedDict
from multiprocessing.pool import Pool
import tqdm
import librosa


# 获取当前脚本的文件路径
current_path = os.path.abspath(__file__)
# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(current_path))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import argparse
import functools
from utils.utils import add_arguments


#  这个是一个命令行工具，输入是jsonl文件，输出是切割好的文件夹。
# 文件夹下有命名规则， speed subj sentence_idx word_idx idx均为相对索引。
# 文件夹下的文件有eeg,music,jsonl一一对应

def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts


def convert_od_to_file_path(od):
    path=''
    # 路径弄完之后还有basename
    for k,v in od.items():
        path+=f'{k}{v}/'
    # basename也要加
    path+=path.replace('/','')
    return path


def get_onsets(raw):
    copy_raw=copy.deepcopy(raw)
    copy_raw.pick(picks='stim', verbose=False)
    copy_raw=copy_raw.load_data( verbose=False)
    eeg_sr=copy_raw.info['sfreq']
    data=copy_raw.get_data(verbose=False)
    onsets=[]
    for i in range(-6,-1):
        onsets.append(np.argmax(data[i])/eeg_sr)
    return onsets


def get_speed_transcription(path):
    path=os.path.join(root_path,path)
    with open(path, 'r', encoding='utf-8') as f:
        transcription = json.load(f)

    for i in range(5):
        speed=i+1
        trans=copy.deepcopy(transcription)
        new_trans={
            "text":''.join([trans['segments'][j]['text'] for j in range(len(trans['segments']))]),
            "segments":[{
                "text":trans['segments'][j]['text'],
                "start":trans['segments'][j]['start']/speed,
                "end":trans['segments'][j]['end']/speed,
                "words":trans['segments'][j]['words']
            } for j in range(len(trans['segments']))]
        }
        new_trans_path=f'transcribe_with_words/{speed}x.json'
        new_trans_path=os.path.join(root_path,new_trans_path)
        makedirs(new_trans_path)
        with open(new_trans_path, "w") as json_file:
            json.dump(new_trans, json_file, indent=4, ensure_ascii=False)

def preprocess_raw(raw:mne.io.Raw):
    # 这个只做选channel和滤波操作
    raw.pick(picks=raw.ch_names[:256], verbose=False)
    raw=raw.load_data(verbose=False)
    # raw.notch_filter(50, verbose=False)
    raw.filter(l_freq=1, h_freq=40, verbose=False)
    raw.resample(target_eeg_sr, verbose=False)
    return raw.get_data()


def get_data_from_subject(subj):
    subj_path=os.path.join(root_path,f'subj{subj}.mff')
    transcribe_dir=os.path.join(root_path,'transcribe_with_words')
    raw = mne.io.read_raw_egi(subj_path, verbose=False, preload=False)
    onsets=get_onsets(raw)
    data=preprocess_raw(raw)
    lines =[]
    for i,onset in enumerate(onsets):
        speed=i+1
        transcribe_path=os.path.join(transcribe_dir,f'{speed}x.json')
        with open(transcribe_path, 'r', encoding='utf-8') as f:
            transcription_dict = json.load(f)
        segments=transcription_dict['segments']

        if subj == 1:
            read_music_path = f'/hpc2hdd/home/yyang937/datasets/music/music x {speed}.mp3'
            wav, wav_sr = sf.read(read_music_path, always_2d=True)
            if wav_sr != 16000:
                wav = wav[:, 0]
                wav = librosa.resample(wav, orig_sr=wav_sr, target_sr=target_sr)
        for j,seg in enumerate(segments):
            start_sec=seg['start']
            end_sec=seg['end']
            eeg_start_sec=onset+start_sec
            eeg_end_sec=onset+end_sec
            eeg_start_index=int(eeg_start_sec*target_eeg_sr)
            eeg_end_index=int(eeg_end_sec*target_eeg_sr)
            print(f'data {data.shape},onset:{onset},start_sec:{start_sec},start:{eeg_start_index},end:{eeg_end_index}')
            eeg_seg=data[:,eeg_start_index:eeg_end_index]
            if eeg_seg.shape[1]==0:
                continue
            eeg_seg_path=os.path.join(root_path,output_data_dir,f'speed_{speed}',f'subj_{subj}',f'seg_{j}.npy')
            speech_seg_path=os.path.join(root_path,output_data_dir,f'speed_{speed}','music',f'seg_{j}.wav')
            makedirs(eeg_seg_path)
            np.save(eeg_seg_path,eeg_seg)

            if subj == 1:
                wav_onset=int(seg['start']*target_sr)
                wav_offset=int(seg['end']*target_sr)
                seg_wav=wav[wav_onset:wav_offset]
                seg_wav_path=os.path.join(root_path,output_data_dir,f'speed_{speed}','music',f'seg_{j}.wav')
                sf.write(makedirs(seg_wav_path),seg_wav,samplerate=target_sr)

            line={
            "speech": {"path": speech_seg_path, 'sr': target_sr},
            "eeg": {"path": eeg_seg_path, 'sr': target_eeg_sr},
            "duration": end_sec-start_sec,
            "sentence": seg['text'].strip(),
            "sentences": [{"text": seg['text'].strip(),
                           "start": 0.0, "end": end_sec-start_sec, "duration": end_sec-start_sec,
                           "words":[{"word": old_word["word"], "start":old_word["start"]-start_sec, "end": old_word["end"]-start_sec}
                                    for old_word in seg['words']]
                           }],
            'subj': subj,
            'speed': speed,
            'language':"Chinese"
            }
            lines.append(line)
    return lines


def get_all_lines():
    subj_list=[i+1 for i in range(20)]
    if use_multiprocessing:
        pool = Pool(processes=20)
        results = pool.map(get_data_from_subject, subj_list)
        pool.close()
        pool.join()
        # print(results[0])
        all_lines1 = []
        for lines in results:
            all_lines1.extend(lines)
    else:
        all_lines1=[]
        for i in tqdm.tqdm(range(20)):
            all_lines1.extend(get_data_from_subject(i+1))
    # print(all_lines1)
    write_jsonlines(
        os.path.join(root_path,output_data_dir, 'all_info.jsonl'), all_lines1)
    return all_lines1


def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path


if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    target_eeg_sr=200
    target_sr=16000
    root_path='/hpc2hdd/home/yyang937/datasets/music'
    get_speed_transcription('/hpc2hdd/home/yyang937/datasets/music/transcribe.json')
    output_data_dir='processed2'
    new_output_data_dir=os.path.join(root_path,output_data_dir,'cut_words')
    use_multiprocessing=True
    # jl=get_all_lines()
    # 读取jsonl文件
    jl=read_jsonlines(os.path.join(root_path,output_data_dir,'all_info.jsonl'))
    output_jl=[]
    for l_idx,line in enumerate(jl):
        # 获取数据
        # 读取EEG 数据
        eeg_line_data=np.load(line['eeg']['path'])
        # print(eeg_line_data.shape,'eeg')
        eeg_sr=line['eeg']['sr']
        subj=line['subj']
        # 读取音频数据
        audio_line_data,audio_sr=sf.read(line['speech']['path'])
        speech_sr=line['speech']['sr']
        speed=line['speed']
        if speed!=1:
            continue
        # 遍历所有句子
        for sent_idx,sent in enumerate(line['sentences']):
            # 遍历所有单词
            for word_idx,word in enumerate(sent['words']):
                word_start_time=word['start']-sent['words'][0]['start']
                word_end_time=word['end']-sent['words'][0]['start']
                word_eeg_start_idx=int(word_start_time*eeg_sr)
                word_eeg_end_idx=int(word_end_time*eeg_sr)
                if word_eeg_start_idx==word_eeg_end_idx:
                    continue
                word_speech_start_idx=int(word_start_time*speech_sr)
                word_speech_end_idx=int(word_end_time*speech_sr)
                # 切割出来
                eeg_word_data=eeg_line_data[:,word_eeg_start_idx:word_eeg_end_idx]
                speech_word_data=audio_line_data[word_speech_start_idx:word_speech_end_idx]

                print(f'eeg length:{eeg_line_data.shape[1]}, eeg_word_data:{eeg_word_data.shape},'
                      f'speed:{speed}'
                      f'word_start_time:{word_start_time},word_end_time:{word_end_time}'
                      f'word_eeg_start_idx:{word_eeg_start_idx},word_eeg_end_idx:{word_eeg_end_idx}')
                # if eeg_word_data.shape[1] == 0:
                #     continue
                # 保存数据
                od=OrderedDict({
                    "speed":speed,
                    "subj":subj,
                    "sent":l_idx,
                    "word":word_idx,
                })

                base_name=convert_od_to_file_path(od)
                seg_meg_path=os.path.join(new_output_data_dir,base_name+'.npy')
                seg_audio_path=os.path.join(new_output_data_dir,base_name+'.wav')
                makedirs(seg_meg_path)
                np.save(seg_meg_path,eeg_word_data)
                sf.write(seg_audio_path,speech_word_data,samplerate=speech_sr)
                word_duration_time=word_end_time-word_start_time
                # 做 jsonl 文件
                word_json={
                    "speech": {"path": seg_audio_path, 'sr': speech_sr},
                    "eeg": {"path": seg_meg_path, 'sr': eeg_sr},
                    'speed':speed,
                    'subj':subj,
                    'language':line['language'],
                    'sentence':word['word'],
                    'duration':word_duration_time,
                    'start':0,
                    'end':word_duration_time,
                    'sentences':[{'text':word['word'],'start':0,'end':word_duration_time,
                                  'duration':word_duration_time}]
                }
                output_jl.append(word_json)
    output_jl_path=os.path.join(new_output_data_dir,'cut_words.jsonl')
    write_jsonlines(output_jl_path,output_jl)

