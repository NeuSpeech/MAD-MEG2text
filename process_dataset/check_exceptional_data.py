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
from multiprocessing import Pool
import numpy as np
import jsonlines
from collections import Counter


def detect_outliers(arr):
    # 判断超过20%的值为0
    if np.count_nonzero(arr == 0) / len(arr) > 0.2:
        return f"{np.count_nonzero(arr == 0) / len(arr)*100}%的值为0"

    # 判断是否存在NaN或None
    if np.shape(arr)[1]==0:
        return "数组长度为0"

    # 判断是否存在NaN或None
    if np.isnan(arr).any() or None in arr:
        return "存在NaN或None"

    # 判断是否存在正负无穷大
    if np.isinf(arr).any():
        return "存在正负无穷大"

    if arr.shape[1]>6000:
        return "数据太长"

    # 可以添加其他异常情况的判断逻辑

    return "正常"

def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts


def detect_outliers_json(json_dict):

    meg=np.load(json_dict['eeg']['path'])
    do=detect_outliers(meg)
    if do!='正常':
        print(do,json_dict)
    return do


def detect_time_json(json_dict):
    start=json_dict['sentences'][0]["start"]
    end=json_dict['sentences'][0]["end"]
    duration=json_dict['sentences'][0]["duration"]
    sl=len(json_dict['sentences'])

    state=[]
    if sl>1:
        state.append('len')
    if start>=30:
        print("start",start,json_dict)
        state.append('start')
    if end>=30:
        print("end",end,json_dict)
        state.append('end')
    if duration>=30:
        print("duration",duration,json_dict)

        state.append('duration')
    return state


if __name__ == '__main__':

    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("jsonl",    type=str, nargs="+", default=[],       help="jsonl文件路径")
    args = parser.parse_args()
    for json in args.jsonl:
        json=os.path.join(home_dir,json)
        all_lines = read_jsonlines(json)
        print(f"检查文件{json}的异常值")
        pool = Pool(processes=32)
        results = pool.map(detect_time_json, all_lines)
        pool.close()
        pool.join()
        results_=[]
        for r in results:
            results_.extend(r)
        count_results = Counter(results_)
        print(count_results)
        print("检查完毕")