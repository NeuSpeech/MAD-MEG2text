from moviepy.editor import VideoFileClip
import sys
import soundfile as sf
import mne
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne.preprocessing import compute_proj_eog, compute_proj_ecg
from mne.preprocessing import maxwell_filter
import numpy as np
import os
import pickle
import json
import traceback
from tqdm import tqdm
import whisper
import torch
import soundfile as sf
import numpy as np
import bypy
import librosa
import jieba
from collections import Counter
import jsonlines
import re
from hanziconv import HanziConv
from scipy.signal import hilbert, butter, filtfilt,firwin
import scipy.signal as sg


def fir_filter(sample, sample_rate, low_cutoff=None, high_cutoff=None):

    nyq=sample_rate/2
    if low_cutoff is not None and high_cutoff is not None:
        cutoff = [low_cutoff/nyq, high_cutoff/nyq]
    elif low_cutoff is None and high_cutoff is not None:
        cutoff=[0,high_cutoff/nyq]
    elif low_cutoff is not None and high_cutoff is None:
        cutoff=[low_cutoff/nyq,(nyq-1e-10)/nyq]
    else:
        raise ValueError("必须指定低通或高通边界")

    ord = int(1200*max(cutoff))
    ord |= 1  # 取奇数

    b = sg.firwin(numtaps=ord,
                  cutoff=cutoff,)

    filtered = sg.lfilter(b, 1, sample)

    return filtered

def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    return path


def torch_random_choices(samples, choices):
    indices = torch.randperm(len(samples))[:choices]
    rand_choices = [samples[i] for i in indices]
    return rand_choices


def segment_video(video_path, store_path, start, end):
    # try:
    clip = VideoFileClip(video_path)
    subclip = clip.subclip(start, end)
    subclip.write_videofile(mkdir(store_path))
    return True
    # except:
    #     return False


def get_segment_from_idx(array, idx):
    # 看视频的时候的开始和结束
    assert len(array) == 32
    start = array[4 * idx, 0]
    end = array[4 * idx + 1, 0]
    return (start, end)


def multi_band_hilbert(signal, sample_rate, bands):
    out = np.zeros((len(bands), signal.shape[0]))
    for i, band in enumerate(bands):
        low, high = band
        for j in range(signal.shape[0]):
            # 使用FIR滤波器滤波信号
            filtered = fir_filter(signal[j, :], sample_rate, low, high)
            # 计算希尔伯特变换的强度平均值
            hilbert_env = np.abs(hilbert(filtered))
            out[i, j] = np.mean(hilbert_env)
    out = out.flatten()
    return out


def adjust_db_float(sound, db=-20):

    assert isinstance(sound, np.ndarray), 'Sound must be np.ndarray'

    max_amp = np.max(np.abs(sound)) # for floating point

    sound *= (10.0 ** (db/20)) / max_amp

    # No type conversion needed
    return np.clip(sound, -1, 1)

def get_audio_segment_from_idx(array, idx):
    # 看视频的时候的开始和结束
    assert len(array) == 32
    start = array[4 * idx + 2, 0]
    end = array[4 * idx + 3, 0]
    return (start, end)


class PreprocessorWithAudio:
    def __init__(self):
        # 整理好的数据有audio,eeg足矣
        self.path = '/home/yyang/dataset/multi_media/main_exp'
        self.watch_path = '/home/yyang/dataset/multi_media/select_videos'
        self.notch_freqs = [50 * (x + 1) for x in range(4)]
        self.set_fix_eye = False
        self.raw = None
        self.seg_info = None
        self.eeg_sr = 1000
        self.eeg_dir = 'EEG_with_audio'

        self.audio_sr = 44100

        self.video_least_duration = 2500  # second
        self.video_output_dir = 'OutputVideo'
        self.duration = None

    def load_data(self, path, preload=True):
        # 读取数据
        # print(path)
        raw = mne.io.read_raw_curry(path, preload=preload, verbose=False)
        self.raw = raw

    def filter_data(self, notch_freqs=None, picks=None, method='spectrum_fit'):
        if notch_freqs is None:
            notch_freqs = self.notch_freqs
        # 使用 notch_filter 函数滤除指定频率的信号
        self.raw.notch_filter(freqs=notch_freqs, picks=picks, method=method, verbose=False)
        self.raw.filter(l_freq=1.0, h_freq=200, verbose=False)

    def fix_bads(self, ):
        self.raw.interpolate_bads(reset_bads=True, verbose=False)

    def fix_eye(self, exclud_ch_names=['HEO', 'VEO']):
        if self.set_fix_eye:
            # 使用 ICA 去除眼电伪迹
            ica = ICA(n_components=34, random_state=42)
            ica.fit(self.raw, verbose=False)
            # 创建EOG epochs
            eog_epochs = mne.preprocessing.create_eog_epochs(self.raw, ch_name=exclud_ch_names, verbose=False)

            # 使用EOG epochs进行ICA降噪并找到相关的ICA成分
            eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name=exclud_ch_names, verbose=False)

            ica.exclude += eog_inds
            ica.apply(self.raw, verbose=False)
            print('ICA SUCCESS')

            # # 应用平均参考电位
            # self.raw.set_eeg_reference(ref_channels=['M1','M2'])
            #
            # # 创建EOG Epochs对象并应用平均参考电位
            # eog_events = mne.preprocessing.find_eog_events(self.raw)
            # eog_epochs = mne.preprocessing.create_eog_epochs(self.raw, events=eog_events, ch_name=exclud_ch_names)
            # # 去除眼电干扰
            # self.raw.subtract(eog_epochs.average())

    def get_info(self, number):
        self.eeg_sr = self.raw.info['sfreq']
        array, _ = mne.events_from_annotations(self.raw, verbose=False)
        self.info = array
        self.events = _
        return array

    def get_video_path(self, number):
        folder_path = os.path.join(self.path, str(number), 'Video')
        file_paths = []
        for file_name in os.listdir(folder_path):
            # 检查文件名后缀是否为.mp4
            if file_name.endswith('.mp4'):
                # 构建完整的文件路径
                file_path = os.path.join(folder_path, file_name)
                file_paths.append(file_path)
        assert len(file_paths) == 1, f'{number}/Video 下应该只有一个视频'
        return file_path

    def check_video(self, path):
        video = VideoFileClip(path)
        # 获取视频的宽度和高度
        width, height = video.size
        # 获取视频的帧率
        fps = video.fps
        self.duration = video.duration
        if self.duration > self.video_least_duration:
            return True
        else:
            return False

    def get_eeg_path(self, number):
        try:
            folder_path = os.path.join(self.path, str(number), 'EEG')
            file_paths = []
            for file_name in os.listdir(folder_path):
                # 检查文件名后缀是否为.cdt
                if file_name.endswith('.cdt'):
                    # 构建完整的文件路径
                    file_path = os.path.join(folder_path, file_name)
                    file_paths.append(file_path)
            # assert len(file_paths) == 1, f'{number}/EEG 下应该只有一个.cdt'
            if len(file_paths) == 1:
                return file_path
        except:
            pass
        return None

    def get_audio_path(self, number):
        folder_path = os.path.join(self.path, str(number), 'Bone_audio')

        # 获取文件夹中的所有文件
        file_list = os.listdir(folder_path)
        # print(file_list)
        # 过滤出以'zoom'开头且以'.wav'结尾的文件
        file_list = [file for file in file_list if file.startswith('ZOOM') and file.endswith('.WAV')]

        # 按文件名的序号进行排序
        sorted_files = sorted(file_list, key=lambda x: int(x[4:-4]))

        # 构建有序的文件地址数组
        file_paths = [os.path.join(folder_path, file) for file in sorted_files]
        return file_paths

    def get_video_output_path(self, number, seg_idx):
        path = os.path.join(self.video_output_dir, str(number), 'watch', f'{seg_idx}.mp4')
        return path

    def into_np(self):
        # 从Raw对象中获取数据
        data = self.raw.get_data(return_times=False, verbose=False)[:64]
        return data  # [ch,time]

    def save_data(self, data_dict, number):
        eeg_path = os.path.join(self.eeg_dir, str(number), 'eeg_data.pkl')
        mkdir(eeg_path)
        with open(eeg_path, 'wb+') as f:
            pickle.dump(data_dict, f)

    def get_watch_video_duration(self, number, seg_idx, return_video_name=False):
        # 88之前一直是火车
        # 89开始是小森林
        if number < 89:
            video_path = {'0': '伤心',
                          '1': '平和1_1',
                          '2': '开心',
                          '3': '愤怒1',
                          '4': '平和',
                          '5': '伤心2',
                          '6': '开心2',
                          '7': '愤怒2',
                          }
        else:
            video_path = {'0': '伤心',
                          '1': '小森林夏秋篇2',
                          '2': '开心',
                          '3': '愤怒1',
                          '4': '小森林夏秋篇1',
                          '5': '伤心2',
                          '6': '开心2',
                          '7': '愤怒2',
                          }
        # seg_idx_video_dict = {key:VideoFileClip(video_path[key]).duration for key in video_path.keys()}
        if return_video_name:
            return VideoFileClip(os.path.join(self.watch_path, video_path[str(seg_idx)]) + '.mp4').duration, video_path[
                f'{seg_idx}']
        else:
            return VideoFileClip(os.path.join(self.watch_path, video_path[str(seg_idx)]) + '.mp4').duration

    def save_subject_data(self, number):
        # 保存eeg数据
        eeg_path = self.get_eeg_path(number)
        if eeg_path is None:
            pass
        else:
            audio_path_list = self.get_audio_path(number)
            self.load_data(eeg_path)
            self.get_info(number)
            # self.filter_data()
            # self.fix_bads()
            # self.fix_eye()
            np_data = self.into_np()
            data_dict = {}
            data_dict['all_data'] = np_data
            data_dict['info'] = self.info
            data_dict['video_sec'] = {}
            # print(audio_path_list,self.info)
            for seg_idx in range(8):
                # 从脑电信号中得到分段的时间
                seg_time_idx_s, seg_time_idx_e = get_audio_segment_from_idx(self.info, seg_idx)
                audio_path = audio_path_list[seg_idx]
                waveform, sample_rate = sf.read(audio_path)
                audio_duration = waveform.shape[0] / sample_rate
                seg_duration = (seg_time_idx_e - seg_time_idx_s) / self.eeg_sr
                print(seg_duration - audio_duration, seg_idx, audio_duration, seg_duration)
                print('fail') if np.abs(seg_duration - audio_duration) > 5 else print('success')
                # seg_data = np_data[:, seg_time_idx_s:int(seg_time_idx_s + audio_duration * self.eeg_sr)]
                # print(seg_time_idx_s / self.eeg_sr, 'start time')
                # data_dict[str(seg_idx)] = seg_data
                # seg_time = watch_video_duration
                # print(number, seg_idx, seg_time, 's')
                # # 对视频分段保存
                # success=segment_video(video_path,self.get_video_output_path(number,seg_idx),*seg_time)
            # self.save_data(data_dict, number)
            # 保存audio气导数据
            return data_dict, audio_path_list

    def __call__(self, number):
        # video_path=self.get_video_path(number)
        eeg_path = self.get_eeg_path(number)
        if eeg_path is None:
            pass
        else:
            self.load_data(eeg_path)
            self.get_info(number)
            self.filter_data()
            self.fix_bads()
            # self.fix_eye()
            np_data = self.into_np()
            data_dict = {}
            data_dict['all_data'] = np_data
            data_dict['info'] = self.info
            data_dict['video_sec'] = {}
            for seg_idx in range(8):
                # 从脑电信号中得到分段的时间
                seg_time_idx_s, seg_time_idx_e = get_segment_from_idx(self.info, seg_idx)
                watch_video_duration = self.get_watch_video_duration(number, seg_idx)
                data_dict['video_sec'][seg_idx] = watch_video_duration
                seg_data = np_data[:, seg_time_idx_s:int(seg_time_idx_s + watch_video_duration * self.eeg_sr)]
                print(seg_time_idx_s / self.eeg_sr, 'start time')
                data_dict[str(seg_idx)] = seg_data
                seg_time = watch_video_duration
                print(number, seg_idx, seg_time, 's')
                # # 对视频分段保存
                # success=segment_video(video_path,self.get_video_output_path(number,seg_idx),*seg_time)
            self.save_data(data_dict, number)


# check audio and eeg validation
def check_audio_eeg(seg_info, audio_path_list):
    seg_secs = seg_info[:, 0] / 1000
    wav_secs = [sf.read(audio_path)[0].shape[0] / 44100 for audio_path in audio_path_list[:-1]]
    # wav_e_secs is the time audio ends
    wav_e_secs = [seg_secs[4 * i + 2] + wav_secs[i] for i, _ in enumerate(wav_secs)]
    wav_e_secs = np.array(wav_e_secs)
    # wav_e1_secs is the time next video starts
    wav_e1_secs = [seg_secs[4 * i + 4] for i, _ in enumerate(wav_secs)]
    wav_e1_secs = np.array(wav_e1_secs)
    confict_sec = wav_e1_secs - wav_e_secs  # should >0 in normal cases
    if np.min(confict_sec) < 0:
        return 0
    else:
        return 1


def stat_word_freq(text):
    words = jieba.lcut(text)  # 使用jieba进行分词
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    return word_freq


def stat_char_freq(text):
    char_freq = {}

    for char in text:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1

    return char_freq


def merge_dicts(dict1, dict2):
    merged_dict = Counter(dict1)
    merged_dict.update(dict2)
    return merged_dict


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts


def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)


# 切分sentence


def combine_words_to_sentences(word_list):
    sentences = []
    current_sentence = ""
    start_time = None
    end_time = None

    # 定义标点符号的正则表达式模式
    punctuation_pattern = r'[,.?!，。？]'

    for word in word_list:
        word['word'] = HanziConv.toSimplified(word['word'])

        if start_time is None:
            start_time = word['start']

        end_time = word['end']

        current_sentence += word['word']
        # 使用正则表达式匹配标点符号
        if re.search(punctuation_pattern, word['word']) or len(word['word']) > 4:
            sentence = {
                'text': current_sentence,
                'start': start_time,
                'end': end_time
            }
            sentences.append(sentence)
            current_sentence = ""
            start_time = None
            end_time = None

    # 处理最后一个句子
    if current_sentence:
        sentence = {
            'text': current_sentence,
            'start': start_time,
            'end': end_time
        }
        sentences.append(sentence)

    return sentences


def combine_words_to_sentences_within_time_limit(word_list, time_limit):
    sentences = []
    current_sentence = ""
    start_time = None
    end_time = None

    # 定义标点符号的正则表达式模式
    punctuation_pattern = r'[,.?!，。？]'
    words=[]
    for word_idx, word in enumerate(word_list):
        word['word'] = HanziConv.toSimplified(word['word'])

        if start_time is None:
            start_time = word['start']

        end_time = word['end']
        if end_time - start_time <= time_limit:
            # 如果加上这个词的总时间少于限制，就把这个词加上。
            # 然后再判断有没有标点符号，如果有标点符号，或者这个词大于4个字，或者下一个词的时间超过限制就结束这个句子。
            current_sentence += word['word']
            words.append(word)
            # 这是一些终止当前语句的条件
            cond1 = re.search(punctuation_pattern, word['word'])  # 在单词中是否有标点符号
            cond2 = len(word['word']) > 4  # 单词过长
            cond3=word_idx + 1 == len(word_list)  # 已经是最后一个词
            if not cond3:
                cond4 = word_list[word_idx + 1]['end'] - start_time > time_limit  # 如果加了下一个词就会超时
            else:
                cond4=False
                # 使用正则表达式匹配标点符号
            if cond1 or cond2 or cond3 or cond4:
                sentence = {
                    'text': current_sentence,
                    'words': words,
                    'start': start_time,
                    'end': end_time
                }
                sentences.append(sentence)
                current_sentence = ""
                words = []
                start_time = None
                end_time = None

    return sentences


def combine_sentences_within_time_limit(sentences, time_limit):
    combinations = []
    current_combination = []
    current_start_time = None
    current_end_time = None

    for sentence in sentences:
        if current_start_time is None:
            current_start_time = sentence['start']

        current_combination.append(sentence)
        current_end_time = sentence['end']

        combination_duration = current_end_time - current_start_time

        if combination_duration > time_limit:
            combinations.append(current_combination[:-1])
            current_combination = [sentence]
            current_start_time = sentence['start']
            current_end_time = sentence['end']

    # 添加最后一个组合
    if current_combination:
        combinations.append(current_combination)

    return combinations


def combine_single_sentences(sentences):
    return [[sentence] for sentence in sentences]


def check_chinese_punctuation(text, chinese_punctuation='[。？]'):
    # 除了末尾是chinese_punctuation中的字符的，如果有符号就变句号，没有符号就加句号。
    # 定义中文句号和问号的正则表达式

    # 检测结尾是否是中文句号或问号
    if re.search(chinese_punctuation + '$', text):
        return text
    else:
        # 检测最后一个字符是否是中文字符
        if re.search('[\u4e00-\u9fa5]$', text):
            # 最后一个字符是中文字符，添加中文句号
            text += '。'
        else:
            # 最后一个字符不是中文字符，替换为中文句号
            text = re.sub('.$', '。', text)

        return text


def has_duplicates(my_list):
    return len(my_list) != len(set(my_list))


def check_repeat_sentences(sentences):
    # whisper有时候会一直重复，这样输出的数据是完全错的。
    # sentences=[{'text':''},{'text':''}]
    texts = [sent['text'] for sent in sentences]
    if has_duplicates(texts):
        return True
    else:
        return False


def contains_english(text):
    pattern = r'[a-zA-Z]'
    match = re.search(pattern, text)
    return match is not None


def has_multiple_english_words(sentence, times=2):
    pattern = r'\b[a-zA-Z]+\b'
    matches = re.findall(pattern, sentence)
    english_words = [word for word in matches if len(word) > 1]
    return len(english_words) > times


def combinations_to_data_dict(combinations):
    data_dict_list = []
    for comb_idx, comb in enumerate(combinations):
        comb[-1]['text'] = check_chinese_punctuation(comb[-1]['text'])
        start = comb[0]['start']
        end = comb[-1]['end']
        text = ''.join([t['text'] for t in comb])
        data_dict_list.append(
            {
                'start': start,
                'end': end,
                'sentence': text,
                'duration': end - start,
                'sentences': comb,
            }
        )
    return data_dict_list


def correct_timing(sentences):
    # let the start timing be 0, other timing should be corrected
    # sentences:[{},{}]
    start = sentences[0]['start']
    for sent_idx, sent in enumerate(sentences):
        sentences[sent_idx]['start'] -= start
        sentences[sent_idx]['end'] -= start
        for word_idx, word in enumerate(sent['words']):
            word['start'] -= start
            word['end'] -= start

    return sentences
