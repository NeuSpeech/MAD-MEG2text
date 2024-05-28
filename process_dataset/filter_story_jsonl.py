import jsonlines
import os
import sys
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

# python process_dataset/filter_story_jsonl.py
if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    val_a_story=True
    # parser = argparse.ArgumentParser(description=__doc__)
    # add_arg = functools.partial(add_arguments, argparser=parser)
    # add_arg("jsonl",    type=str, default=None,       help="jsonl文件路径")
    # add_arg("output_dir",    type=str,  default=None,       help="输出jsonl文件夹")
    # args = parser.parse_args()
    input_jsonl='datasets/gwilliams2023/preprocess7/info.jsonl'
    output_dir='datasets/gwilliams2023/preprocess7/split3'
    datas = read_jsonlines(os.path.join(home_dir,input_jsonl))
    story_list=['easy_money', 'cable_spool_fort', 'The_Black_Willow'.lower(), 'lw1']
    # 分割四种
    for hold_out_story in story_list:
        if val_a_story:
            for val_story in [s for s in story_list if s not in [hold_out_story]]:
                train_list=[data for data in datas if data['story'] not in [val_story,hold_out_story]]
                val_list=[data for data in datas if data['story']==val_story]
                test_list=[data for data in datas if data['story']==hold_out_story]

                data_dict = {
                    'train': train_list,
                    'val': val_list,
                    'test': test_list,
                }
                print(val_story,hold_out_story,len(train_list),len(val_list),len(test_list))
                for k,v in data_dict.items():
                    json=os.path.join(home_dir,output_dir,hold_out_story,val_story,f'{k}.jsonl')
                    write_jsonlines(makedirs(json), v)


        else:
            train_val_list=[data for data in datas if data['story']!=hold_out_story]
            test_list=[data for data in datas if data['story']==hold_out_story]
            split_num=int(len(train_val_list)/9*8)
            train_list=train_val_list[:split_num]
            val_list=train_val_list[split_num:]
            print(len(train_list),len(val_list),len(test_list))
            data_dict={
                'train':train_list,
                'val':val_list,
                'test':test_list,
            }
            for mode in ['train','val','test']:
                json=os.path.join(home_dir,output_dir,hold_out_story,f'{mode}.jsonl')
                write_jsonlines(makedirs(json), data_dict[mode])

