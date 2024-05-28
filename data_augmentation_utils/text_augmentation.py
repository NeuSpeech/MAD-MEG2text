# this is used to augment the text data.
# including using GPT and traditional methods
# such as random delete, add, or replace,
# in char-level, word-level and sentence-level.


import os
import json
from openai import OpenAI
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import httpx


class GPTTextEnhancer:
    def __init__(self, client='config/key.json', model='gpt-3.5-turbo', resp_kwargs=None,mid_lang='german'):
        self.client = self.load_client(client)
        self.model = model
        self.history=[]
        self.allowed_types=['Paraphrase','BackTranslation','Add','Extraction']
        self.resp_kwargs = self.load_resp_kwargs(resp_kwargs)  # 这个是获取text恢复时可以传入的参数
        self.mid_lang=mid_lang
        self.aug_type=None

    def load_client(self, client):
        if type(client) == str:  # 加载的config
            with open(client, 'r') as f:
                key = json.load(f)
                key = key['openai_config']
            client = OpenAI(**key,
                            http_client=httpx.Client(
                                base_url="https://api.chatgptid.net/v1",
                                follow_redirects=True,
                            ),
                            )
        return client

    def load_resp_kwargs(self, resp_kwargs):
        if resp_kwargs is None:
            resp_kwargs = {}
        self.resp_kwargs = resp_kwargs
        return resp_kwargs

    def get_openai_response(self, content, role="user", history=None, model=None,**resp_kwargs):
        if model is None:
            model = self.model
        if history is None:
            history = self.history

        # 以后来传入的参数为准
        for key in self.resp_kwargs.keys():
            if key not in resp_kwargs.keys():
                resp_kwargs[key]=self.resp_kwargs[key]
        response = self.client.chat.completions.create(
            model=model,
            messages=[*history, {"role": role, "content": content}],
            stream=False, **resp_kwargs
        )
        return [choice.message.content for choice in response.choices]


    @staticmethod
    def content_to_message(content, role='user'):
        return {"role": role, "content": content}

    def paraphrase(self, content,n):
        prompt=('Now you are writing expert.'
                'Please paraphrase the user input text.'
                'You can vary the text length a little bit, '
                'but you must keep the semantic meaning, and'
                'only return the paraphrased text.')
        history=[self.content_to_message(prompt, 'system')]
        response = self.get_openai_response(content,history=history, role='user',n=n)
        return response

    def translate(self, content,lang,add_prompt='',**kwargs):
        prompt=('Now you are translation expert.'
                f'Please translate the user input text into {lang}.'
                'You should only return translated text.')+add_prompt
        history=[self.content_to_message(prompt, 'system')]
        response = self.get_openai_response(content,history=history, role='user',**kwargs)
        return response

    def detect_language(self, content,**kwargs):
        prompt=('Now you are language expert.'
                f'Please only output the language of users text.')
        history=[self.content_to_message(prompt, 'system')]
        response = self.get_openai_response(content,history=history, role='user',**kwargs)
        return response

    # def back_translation(self, content,n):
    #     mid_lang=self.mid_lang
    #     mid_text=self.translate(content,lang=mid_lang,n=1)[0]
    #     orig_lang=self.detect_language(content)
    #     response = self.translate(mid_text,lang=orig_lang,n=n,
    #                               add_prompt='You are augmenting text, '
    #                                          'you should translate like no one will say the same.')
    #
    #     return response

    def back_translation(self, content, n):
        mid_lang=self.mid_lang
        input_lang=self.detect_language(content)[0]
        prompt=(f"Translate the following text into {mid_lang} and "
                f"then translate the result back into {input_lang}. "
                f"You should output the intermediate result and final back translation. "
                f"You must separate these two text with $sep$."
                f"You are not allowed to output irrelevant context such \\n."
                f"Quote sign must be close to text without any blank space.")
        history=[self.content_to_message(prompt, 'system')]
        response = self.get_openai_response(content,history=history, role='user', n=n)
        response = [text.split('$sep$')[-1].replace('  ','') for text in response]

        return response

    def add(self, content,n):
        prompt=('Now you are writing expert.'
                f'You are asked to add some details to the user text.'
                f'You must keep the semantic meaning.'
                f'You should only return your modified text.')
        history=[self.content_to_message(prompt, 'system')]
        response = self.get_openai_response(content,history=history, role='user',n=n)
        return response

    def extract(self, content,n):
        prompt=('Now you are writing expert.'
                f'You are asked to extract important words in the user text.'
                f'You must keep the basic semantic meaning.'
                f'You should only return your modified text.')
        history=[self.content_to_message(prompt, 'system')]
        response = self.get_openai_response(content,history=history, role='user',n=n)
        return response

    def augment(self,content,n=1, num_thread=1):
        if self.aug_type=='Paraphrase':
            return self.paraphrase(content,n)
        elif self.aug_type=='BackTranslation':
            return self.back_translation(content,n)
        elif self.aug_type=='Add':
            return self.add(content,n)
        elif self.aug_type=='Extraction':
            return self.extract(content,n)
        else:
            raise NotImplementedError


class TextAug:
    def __init__(self,client='data_augmentation_utils/config/key.json', model='gpt-3.5-turbo', resp_kwargs=None):
        self.gpt_enhancer=GPTTextEnhancer(client=client, model=model, resp_kwargs=resp_kwargs)
        self.funcs=[
            'KeyBoard',
            'Ocr',
            'RandomCharInsert',
            'RandomCharSubstitute',
            'RandomCharSwap',
            'RandomCharDelete',
            'Antonym',
            'Contextual',
            'RandomWordCrop',
            'RandomWordSubstitute',
            'RandomWordSwap',
            'RandomWordDelete',
            'Spelling',
            'Split',
            'Synonym',
            'TfIdf',
            'WordEmbs',
            'Paraphrase',
            'BackTranslation',
            'Add',
            'Extraction'
        ]

    def __call__(self, text, func, init_kwargs={}, n=1, num_thread=1):
        assert type(text) is str
        assert func in self.funcs, f'{func} is not in allowed types {self.funcs}'
        if func == 'KeyBoard':
            aug=nac.KeyboardAug(aug_char_p=0.1,**init_kwargs)
        elif func == 'Ocr':
            aug=nac.OcrAug(**init_kwargs)
        elif func == 'RandomCharInsert':
            aug=nac.RandomCharAug(action="insert",aug_char_p=0.1,**init_kwargs)
        elif func == 'RandomCharSubstitute':
            aug=nac.RandomCharAug(action="substitute",aug_char_p=0.1,**init_kwargs)
        elif func == 'RandomCharSwap':
            aug=nac.RandomCharAug(action="swap",aug_char_p=0.1,**init_kwargs)
        elif func == 'RandomCharDelete':
            aug=nac.RandomCharAug(action="delete",aug_char_p=0.1,**init_kwargs)
        elif func == 'Antonym':
            aug=naw.AntonymAug(aug_p=0.1,**init_kwargs)
        elif func == 'Contextual':
            aug=naw.ContextualWordEmbsAug(action="substitute",device='cuda',model_type='bert',**init_kwargs)
        elif func == 'Synonym':
            aug=naw.SynonymAug(**init_kwargs)
        elif func == 'RandomWordCrop':
            aug=naw.RandomWordAug(action="crop",**init_kwargs)
        elif func == 'RandomWordSubstitute':
            aug=naw.RandomWordAug(action="substitute",**init_kwargs)
        elif func == 'RandomWordSwap':
            aug=naw.RandomWordAug(action="swap",**init_kwargs)
        elif func == 'RandomWordDelete':
            aug=naw.RandomWordAug(action="delete",**init_kwargs)
        elif func == 'Spelling':
            aug=naw.SpellingAug(**init_kwargs)
        elif func == 'Split':
            aug=naw.SplitAug(**init_kwargs)
        elif func == 'Synonym':
            aug=naw.SynonymAug(**init_kwargs)
        elif func == 'TfIdf':
            aug=naw.TfIdfAug(**init_kwargs)
        elif func == 'WordEmbs':
            aug=naw.WordEmbsAug(model_type='word2vec',
                                model_path='word2vec/GoogleNews-vectors-negative300.bin',
                                **init_kwargs)
        elif func in self.gpt_enhancer.allowed_types:
            self.gpt_enhancer.aug_type=func
            aug=self.gpt_enhancer
        else:
            raise NotImplementedError
        # print(len(text),len(text.split(' ')),text)
        text=aug.augment(text,n,num_thread)
        # print(text)
        return text




