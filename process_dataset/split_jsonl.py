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



if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    val_a_story=True
    # parser = argparse.ArgumentParser(description=__doc__)
    # add_arg = functools.partial(add_arguments, argparser=parser)
    # add_arg("jsonl",    type=str, default=None,       help="jsonl文件路径")
    # add_arg("output_dir",    type=str,  default=None,       help="输出jsonl文件夹")
    # args = parser.parse_args()
    input_jsonl='datasets/music/processed2/cut_words/cut_words.jsonl'
    output_dir=os.path.join(home_dir,'datasets/music/processed2/cut_words')
    all_lines1 = read_jsonlines(os.path.join(home_dir,input_jsonl))
    np.random.shuffle(all_lines1)  # 随机打乱数据列表

    total_samples = len(all_lines1)
    train_samples = int(0.8 * total_samples)
    val_samples = int(0.1 * total_samples)
    test_samples = total_samples - train_samples - val_samples

    train_data1 = all_lines1[:train_samples]
    val_data1 = all_lines1[train_samples:train_samples + val_samples]
    test_data1 = all_lines1[train_samples + val_samples:]

    print("训练集大小:", len(train_data1))
    print("验证集大小:", len(val_data1))
    print("测试集大小:", len(test_data1))
    split1_path = os.path.join(output_dir, 'split1')
    os.makedirs(split1_path, exist_ok=True)
    write_jsonlines(os.path.join(split1_path, 'train.jsonl'), train_data1)
    write_jsonlines(os.path.join(split1_path, 'val.jsonl'), val_data1)
    write_jsonlines(os.path.join(split1_path, 'test.jsonl'), test_data1)
