import sys
import os
current_path = os.path.abspath(__file__)
# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(current_path))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

from data_augmentation_utils.rectify_dataset import TextEnhancer
import os
import argparse
import functools
from utils.utils import add_arguments


# python process_dataset/rectify_dataset.py --jsonl='datasets/gwilliams2023/preprocess5/info.jsonl' --model='gpt-3.5-turbo'
if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("jsonl",    type=str,  default=None,       help="需要修正的jsonl文件路径")
    add_arg("model",    type=str,  default=None,       help="使用的openai模型类型")
    args = parser.parse_args()
    enhancer=TextEnhancer(client='data_augmentation_utils/config/key.json',model=args.model)
    enhancer.rectify_dataset(jsonl_path=os.path.join(home_dir,args.jsonl))