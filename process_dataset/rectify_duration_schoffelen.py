import jsonlines
import os
import numpy as np
import sys
from multiprocessing import Pool
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


def rectify_time_json(json_dict):
    assert len(json_dict['sentences'])==1
    meg=np.load(json_dict['eeg']['path'])
    meg_duration=meg.shape[1]/json_dict['eeg']['sr']
    json_dict["duration"]=meg_duration
    json_dict['sentences'][0]["end"]=meg_duration
    json_dict['sentences'][0]["duration"]=meg_duration
    return json_dict



if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("jsonl",    type=str, nargs="+", default=[],       help="jsonl文件路径")
    args = parser.parse_args()
    for json in args.jsonl:
        json=os.path.join(home_dir,json)
        datas = read_jsonlines(json)
        pool = Pool(processes=32)
        results = pool.map(rectify_time_json, datas)
        pool.close()
        pool.join()
        write_jsonlines(json, results)

