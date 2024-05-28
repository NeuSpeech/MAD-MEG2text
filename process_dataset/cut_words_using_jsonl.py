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
# 文件夹下有命名规则， subj sentence_idx word_idx idx均为相对索引。
# 文件夹下的文件有eeg audio,jsonl一一对应

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


def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path


def cut_jsonl(jl_path):
    abs_jl_path=os.path.join(home_dir,jl_path)
    jl=read_jsonlines(abs_jl_path)
    output_jl=[]
    new_output_data_dir=abs_jl_path[:-6]+f"{add_sec}_cut_words"
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
        # 遍历所有句子
        for sent_idx,sent in enumerate(line['sentences']):
            # 遍历所有单词
            for word_idx,word in enumerate(sent['words']):
                word_start_time=word['start']-sent['words'][0]['start']
                word_end_time=word['end']-sent['words'][0]['start'] + add_sec
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
                      f'word_start_time:{word_start_time},word_end_time:{word_end_time}'
                      f'word_eeg_start_idx:{word_eeg_start_idx},word_eeg_end_idx:{word_eeg_end_idx}'
                      f'word:{word["word"]}')
                # if eeg_word_data.shape[1] == 0:
                #     continue
                # 保存数据
                od=OrderedDict({
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



if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("jsonl",    type=str, nargs="+", default=[],       help="jsonl文件路径")
    args = parser.parse_args()
    add_sec=0.5
    use_multiprocessing=True
    # 读取 jsonl 文件
    for path in args.jsonl:
        cut_jsonl(path)
