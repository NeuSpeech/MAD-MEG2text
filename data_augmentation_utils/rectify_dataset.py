# 修复文本
# 使用GPT进行文本优化。
# gwilliam的数据集需要优化，每一句都是机翻
# schoffelen数据集的文本不需要优化，因为是每一句都已经固定好了。
# it turns out rectifying text of original text is useless,
# in some cases, the original text is not a whole sentence just as speech.
# so, use GPT to do this, will just add some words to fix the sentence,
# but these words are not in the original speech.
# This means, this way is not useful.

# character.py
import json
import yaml
import os
import sys
import asyncio
import pandas as pd
from openai import OpenAI
from utils.process_utils import read_jsonlines,write_jsonlines

# datasets/schoffelen2019n/preprocess6/audio_info.jsonl
# datasets/gwilliams2023/preprocess5/info.jsonl
def get_sentences(jsonl_path):
    jsonl=read_jsonlines(jsonl_path)
    text_set=set()
    for j in jsonl:
        text=j['sentence']
        text_set.update(text)
    return text_set

def read_txt(path):
    lines = []
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines


class TextEnhancer:
    def __init__(self, client, model='gpt-3.5-turbo', resp_kwargs=None, save_path=None):
        self.client = self.load_client(client)
        self.model = model
        self.save_path = save_path
        self.history=[]
        self.resp_kwargs = self.load_resp_kwargs(resp_kwargs)  # 这个是获取text恢复时可以传入的参数

    def load_client(self, client):
        if type(client) == str:  # 加载的config
            with open(client, 'r') as f:
                key = json.load(f)
                key = key['openai_config']
            client = OpenAI(**key)
        return client

    def load_resp_kwargs(self, resp_kwargs):
        if resp_kwargs is None:
            resp_kwargs = {}
        self.resp_kwargs = resp_kwargs
        return resp_kwargs

    def get_openai_response(self, content, role="user", history=None, model=None):
        if model is None:
            model = self.model
        if history is None:
            history = self.history
        stream = self.client.chat.completions.create(
            model=model,
            messages=[*history, {"role": role, "content": content}],
            stream=True, **self.resp_kwargs
        )
        return stream

    def stream_to_message(self, stream):
        reply = ""
        role = None
        for i, chunk in enumerate(stream):
            if i == 0:
                role = chunk.choices[0].delta.role
            if chunk.choices[0].delta.content is not None:
                reply += chunk.choices[0].delta.content
        assert role is not None, 'role cannot be none, which indicates there is no response'
        return self.content_to_message(content=reply, role=role)

    @staticmethod
    def content_to_message(content, role='user'):
        return {"role": role, "content": content}

    def rectify(self, content,full_correct_content):
        prompt=('Now you are rectifying sentences that is transcribed from speech using deep models, '
                'I will give you the whole text, and user will give you one sentence.'
                'You are required to rectify the user sentence by looking at the whole text and '
                'choosing most appropriate sentence to rectify the user sentence.'
                'Please refer to the whole text as correct version and only return the rectified user sentence.'
                f'The whole text is {full_correct_content}')
        history=[self.content_to_message(prompt, 'system')]
        stream = self.get_openai_response(content,history=history, role='user')

        rectified_text = self.stream_to_message(stream)['content']
        return rectified_text

    def rectify_dataset(self, jsonl_path):
        jsonl=read_jsonlines(jsonl_path)
        story_list=[]
        sentence_list=[]
        for j in jsonl:
            story=j['story'].lower()
            sentence=j['sentence']
            if sentence not in sentence_list:
                story_list.append(story)
                sentence_list.append(sentence)
        story_sentence_list=[(story,sentence) for story,sentence in zip(story_list,sentence_list)]
        story_path_list={
            'cable_spool_fort':'datasets/gwilliams2023/download/stimuli/text/cable_spool_fort.txt',
            'easy_money':'datasets/gwilliams2023/download/stimuli/text/easy_money.txt',
            'lw1':'datasets/gwilliams2023/download/stimuli/text/lw1.txt',
            'the_black_willow':'datasets/gwilliams2023/download/stimuli/text/the_black_willow.txt',
        }
        # 根据故事保存所有的句子
        for story in story_path_list.keys():
            sentence_list_in_story=[ss[1] for ss in story_sentence_list if ss[0] == story]
            story_path=os.path.join(os.path.dirname(jsonl_path),f'story_{story}.yaml')
            with open(story_path, 'w') as f:
                yaml.dump(sentence_list_in_story, f)
        text_rectification_dict={}
        text_story_dict={}
        home_dir = os.path.expanduser("~")
        count=0
        print(f'总共有{len(story_sentence_list)}句')
        for (story,sentence) in story_sentence_list:
            story_text=read_txt(os.path.join(home_dir,story_path_list[story]))[0]
            rectified_text=self.rectify(sentence,story_text)
            text_rectification_dict[sentence]=rectified_text
            text_story_dict[sentence]=story
            count+=1
            print(count)
            print(sentence)
            print(rectified_text)
            print('#'*50)
            # if count>5:
            #     break
            # else:
            #     print(rectified_text)

        # 保存结果

        csv_path=os.path.join(os.path.dirname(jsonl_path),'text_rectification_dict.csv')
        text_rectification_dict_keys=list(text_rectification_dict.keys())
        csv_data={
            'original':[s for s in text_rectification_dict_keys],
            'rectified':[text_rectification_dict[s] for s in text_rectification_dict_keys],
            'story':[text_story_dict[s] for s in text_rectification_dict_keys],
        }
        df=pd.DataFrame(csv_data)
        df.to_csv(csv_path)
        for j in jsonl:
            if j['sentence'] in text_rectification_dict.keys():
                j['sentence']=text_rectification_dict[j['sentence']]
                for sent in j['sentences']:
                    sent['text']=j['sentence']
            else:
                pass
        new_jsonl_path=os.path.join(os.path.dirname(jsonl_path),'rectified_'+os.path.basename(jsonl_path))
        write_jsonlines(new_jsonl_path,jsonl)
        return text_rectification_dict









