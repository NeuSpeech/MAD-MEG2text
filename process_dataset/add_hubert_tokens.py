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
# add hubert token ids in the json
# 获取当前脚本的文件路径
current_path = os.path.abspath(__file__)
# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(current_path))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import argparse
import functools
from utils.utils import add_arguments
from utils.hubert import HubertTokenizer

from multiprocessing import Pool, Manager

def process_line(l_idx, line):
    hubert_tokenizer=HubertTokenizer(hubert_path=hubert_path,
                                 hubert_layer=6,
                                 km_path=km_path,batch_size=batch_size)
    speech_path = line['speech']['path']
    codes = hubert_tokenizer.wav2code(speech_path)
    line['speech']['hubert_codes'] = codes
    return l_idx, line

def collect_result(result):
    global output_jl
    lines,start_id = result
    output_jl[start_id:start_id+len(lines)] = lines
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


def add_hubert_codes_to_jsonl(jl_path):
    hubert_tokenizer=HubertTokenizer(hubert_path=hubert_path,
                                 hubert_layer=6,
                                 km_path=km_path,batch_size=batch_size)
    abs_jl_path=os.path.join(home_dir,jl_path)
    jl=read_jsonlines(abs_jl_path)
    output_jl=[]
    new_output_data_dir=abs_jl_path[:-6]
    use_batch=False
    if use_batch:
        # slower than normal..........
        speech_paths=[]
        all_codes=[]
        for l_idx,line in tqdm.tqdm(enumerate(jl), total=len(jl)):
            speech_path=line['speech']['path']
            speech_paths.append(speech_path)
            if len(speech_paths)==batch_size or l_idx==len(jl)-1:
                codes=hubert_tokenizer.wavs2code(speech_paths)
                all_codes.extend(codes)
                speech_paths=[]

        for l_idx, line in tqdm.tqdm(enumerate(jl), total=len(jl)):
            line['speech']['hubert_codes']=all_codes[l_idx]
            output_jl.append(line)
    else:
        for l_idx, line in tqdm.tqdm(enumerate(jl), total=len(jl)):
            speech_path = line['speech']['path']
            codes = hubert_tokenizer.wav2code(speech_path)
            line['speech']['hubert_codes'] = codes
            output_jl.append(line)
    os.makedirs(new_output_data_dir,exist_ok=True)
    output_jl_path=os.path.join(new_output_data_dir,'add_hubert_codes.jsonl')
    write_jsonlines(output_jl_path,output_jl)


def process_chunk(chunk,start_id,):
    hubert_tokenizer=HubertTokenizer(hubert_path=hubert_path,
                                 hubert_layer=6,
                                 km_path=km_path,batch_size=batch_size)
    lines=[]
    for line in chunk:
        speech_path = line['speech']['path']
        codes = hubert_tokenizer.wav2code(speech_path)
        line['speech']['hubert_codes'] = codes
        lines.append(line)
    return lines,start_id


def add_hubert_codes_to_jsonl_mp(jl_path):
    abs_jl_path = os.path.join(home_dir, jl_path)
    jl=read_jsonlines(abs_jl_path)

    new_output_data_dir = abs_jl_path[:-6]
    output_jl = [None] * len(jl)  # Pre-allocate list for output lines

    # Create a manager for sharing data between processes
    manager = Manager()
    output_jl = manager.list(output_jl)

    # Create a process pool with 32 workers
    pool = Pool(processes=processes)
    chunk_size = len(jl) // processes
    chunks = [jl[i * chunk_size: (i + 1) * chunk_size] if i<processes-1 else jl[i * chunk_size:]  for i in range(processes)]

    # Use imap_unordered to maintain the order of results
    for c_idx, chunk in tqdm.tqdm(enumerate(chunks),total=len(chunks)):
        pool.apply_async(process_chunk, args=(chunk,c_idx * chunk_size), callback=collect_result)

    # Close the pool and wait for all tasks to complete
    pool.close()
    pool.join()

    # Convert the shared list back to a regular list
    output_jl = list(output_jl)
    output_jl_path=os.path.join(new_output_data_dir,'add_hubert_codes.jsonl')
    write_jsonlines(output_jl_path,output_jl)


if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("jsonl",    type=str, nargs="+", default=[],       help="jsonl文件路径")
    args = parser.parse_args()
    hubert_path='download_models/hubert/hubert_base_ls960.pt'
    km_path='download_models/hubert/km200_km.bin'
    processes = 32
    batch_size = 256
    use_mp = False
    hubert_path=os.path.join(home_dir,hubert_path)
    km_path=os.path.join(home_dir,km_path)
    print(hubert_path)
    # 读取 jsonl 文件
    print(args.jsonl)
    for path in args.jsonl:
        print(f'processing {path}')
        if use_mp:
            add_hubert_codes_to_jsonl_mp(path)
        else:
            add_hubert_codes_to_jsonl(path)
