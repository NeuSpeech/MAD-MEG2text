import jsonlines
import os
import sys
import numpy as np
# 这个是在GWilliams数据集里面要过滤某个story
# 获取当前脚本的文件路径
# current_path = os.path.abspath(__file__)
# # 获取项目根目录的路径
# project_root = os.path.dirname(os.path.dirname(current_path))
# # 将项目根目录添加到 sys.path
# sys.path.append(project_root)
#
# import argparse
# import functools
# from utils.utils import add_arguments


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


def split_list(l,):
    ll=len(l)
    split_1=int(0.8*ll)
    split_2=int(0.9*ll)
    return {'train':l[:split_1], 'val':l[split_1:split_2], 'test':l[split_2:]}


# python process_dataset/filter_sentence_jsonl.py
if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    # parser = argparse.ArgumentParser(description=__doc__)
    # add_arg = functools.partial(add_arguments, argparser=parser)
    # add_arg("jsonl",    type=str, default=None,       help="jsonl文件路径")
    # add_arg("output_dir",    type=str,  default=None,       help="输出jsonl文件夹")
    # args = parser.parse_args()
    input_jsonl='datasets/gwilliams2023/preprocess6/info.jsonl'
    output_dir='datasets/gwilliams2023/preprocess6/split4'
    mix_train_val=True
    datas = read_jsonlines(os.path.join(home_dir,input_jsonl))
    story_list=['easy_money', 'cable_spool_fort', 'The_Black_Willow', 'lw1']
    train_data_list=[]
    val_data_list=[]
    test_data_list=[]
    for s in story_list:
        seq_ids=[]
        story_data=[data for data in datas if data['story'].lower() ==s.lower()]
        for i,d in enumerate(story_data):
            seq_ids.append(d["seq_id"])
        seq_ids_set=set(seq_ids)
        # print(seq_ids_set)
        seq_ids=list(seq_ids_set)
        np.random.shuffle(seq_ids)
        seq_ids_dict=split_list(seq_ids)
        print(seq_ids_dict)
        for mode,mode_seq_ids in seq_ids_dict.items():
            # 筛选出seq ids
            mode_data=[md for md in story_data if md['seq_id'] in mode_seq_ids]
            if mode=='train':
                train_data_list.extend(mode_data)
            elif mode=='val':
                val_data_list.extend(mode_data)
            else:
                test_data_list.extend(mode_data)
    if mix_train_val:
        train_val_list=train_data_list+val_data_list
        split_num=len(train_data_list)
        np.random.shuffle(train_val_list)
        train_data_list,val_data_list=train_val_list[:split_num],train_val_list[split_num:]
    data_dict = {
        'train': train_data_list,
        'val': val_data_list,
        'test': test_data_list,
    }
    for mode in ['train', 'val', 'test']:
        json = os.path.join(home_dir, output_dir, f'{"mix_train_val/" if mix_train_val else ""}{mode}.jsonl')
        write_jsonlines(makedirs(json), data_dict[mode])
