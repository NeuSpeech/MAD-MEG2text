import jsonlines
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# 获取当前脚本的文件路径
current_path = os.path.abspath(__file__)
# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(current_path))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import argparse
import functools
from utils.utils import add_arguments


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

def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path

if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    # parser = argparse.ArgumentParser(description=__doc__)
    # add_arg = functools.partial(add_arguments, argparser=parser)
    # add_arg("jsonl_dir",    type=str, default=None,       help="jsonl文件路径")
    # args = parser.parse_args()
    replace_folder = 'preprocess6'
    folder_path = '/hpc2hdd/home/yyang937/datasets/gwilliams2023/download/'
    audio_folder_path = f'/hpc2hdd/home/yyang937/datasets/gwilliams2023/{replace_folder}/audio'
    transcription_dir=os.path.join(audio_folder_path,'transcription')
    transcription_files=os.listdir(transcription_dir)
    transcription_files=[os.path.join(transcription_dir,file) for file in transcription_files if file.endswith('.jsonl')]
    duration_list=[]
    for file_path in transcription_files:
        with open(file_path, 'r') as f:
            file = json.load(f)
        for i,seg in enumerate(file['segments']):
            duration_list.append(seg['end']-seg['start'])
    plt.hist(duration_list, bins=10, alpha=0.7, color='blue', edgecolor='black')
    # 添加标题和标签
    plt.title('Histogram of Data Distribution')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.savefig(makedirs('tmp_fig/duration.pdf'))


