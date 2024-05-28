import os
import sys
# 获取当前脚本的文件路径
current_path = os.path.abspath(__file__)
# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(current_path))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import argparse
import functools
from utils.utils import add_arguments
import json
import yaml
from data_augmentation_utils.text_augmentation import TextAug
from tqdm import tqdm as tbar
from multiprocessing import Pool
import numpy as np
import jsonlines
from collections import Counter


def read_json(file_path):
    with open(file_path,'r') as f:
        file=json.load(f)
    return file



if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("transcription_dir",    type=str, default=None,       help="转录文件路径")
    add_arg("augment_type",    type=str, default=None,       help="文本增强类型")
    add_arg("n",    type=int, default=5,       help="文本增强数量")
    add_arg("save_dir",    type=str, default=None,       help="文件保存路径")
    args = parser.parse_args()
    # 转录文件，然后每种文本增强类型保存一个字典，
    # key就是原text，value就是处理后的text list
    # 1，读取所有文件得到原text list
    # 2，将text处理，变为字典
    original_text_list=[]
    transcription_files=os.listdir(os.path.join(home_dir,args.transcription_dir))
    transcription_files=[os.path.join(home_dir,args.transcription_dir,file)
                         for file in transcription_files if file.endswith('.json')]
    for file in transcription_files:
        data=read_json(file)
        original_text_list.extend([segment['text'] for segment in data['segments']])

    augmenter=TextAug()
    augment_types=[]
    if args.augment_type is None:
        augment_types=augmenter.funcs
    else:
        assert args.augment_type in augmenter.funcs
        augment_types=[args.augment_type]
    for aug_type in augment_types:
        augment_dict={}
        print(aug_type)
        for original_text in tbar(original_text_list):
            if len(original_text.split(' '))<=4 and aug_type=='RandomWordCrop':
                # 如果少于4个单词，不能做RandomWordCrop
                continue
            augmented_text_list=augmenter(text=original_text,aug_type=aug_type,n=args.n)
            augment_dict[original_text]=augmented_text_list
        # 保存下来
        save_path=os.path.join(home_dir,args.save_dir,aug_type+'.yaml')
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(augment_dict, f,allow_unicode=True)
            f.close()





