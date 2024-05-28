import os
from utils.process_utils import read_jsonlines

# datasets/schoffelen2019n/preprocess6/audio_info.jsonl
# datasets/gwilliams2023/preprocess5/info.jsonl
def get_sentences(jsonl_path):
    jsonl=read_jsonlines(jsonl_path)
    text_set=set()
    for j in jsonl:
        text=j['sentence']
        text_set.update(text)
    return text_set
