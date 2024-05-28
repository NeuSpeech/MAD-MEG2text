import jsonlines
import os
import sys
import numpy as np
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
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("jsonl_dir",    type=str, default=None,       help="jsonl文件路径")
    args = parser.parse_args()
    modes=["train","val","test"]
    jsonl_path_list=[args.jsonl_dir+f"{mode}.jsonl" for mode in modes]
    modes_sentences={}
    modes_words={}
    for mode in modes:
        json=os.path.join(home_dir,args.jsonl_dir+f"/{mode}.jsonl")
        datas = read_jsonlines(json)
        sentences=[line['sentence'] for line in datas]
        words=[word for sentence in sentences for word in sentence.split()]
        unique_sentences=set(sentences)
        modes_sentences[mode]=unique_sentences
        modes_words[mode]=set(words)

        print(f'mode:{mode} sentences:{len(sentences)} '
              f'unique sentences:{len(unique_sentences)} '
              f'words:{len(words)} '
              f'unique words:{len(modes_words[mode])} '
              f'\n')
    # 1，要看test set里面与训练集重合的句子数目以及比例
    overlapping_sentences = len(modes_sentences['train'] & modes_sentences['test'])
    all_unique_sentences = len(modes_sentences['train'] | modes_sentences['val'] | modes_sentences['test'])
    all_unique_words = len(modes_words['train'] | modes_words['val'] | modes_words['test'])
    overlapping_ratio = overlapping_sentences / len(modes_sentences['test'])
    # 2，要看test set里面的单词与训练集重合的数目以及比例
    all_train_words = modes_words['train']
    all_test_words = modes_words['test']
    overlapping_words = len(all_train_words & all_test_words)
    overlapping_word_ratio = overlapping_words / len(all_test_words)
    # 想要得到的信息,每个mode下的句子数，独特句子数，单词数，独特单词数。
    # 测试集和训练集重合的单词数以及比例
    print(f'all_unique_sentences: {all_unique_sentences}')
    print(f'all_unique_words: {all_unique_words}')
    print(f'Overlapping sentences: {overlapping_sentences}')
    print(f'Overlapping sentence ratio: {overlapping_ratio}')
    print(f'Overlapping words: {overlapping_words}')
    print(f'Overlapping word ratio: {overlapping_word_ratio}')
