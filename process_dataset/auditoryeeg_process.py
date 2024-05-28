import gzip
import numpy as np
import soundfile as sf
import librosa
import os
import pandas as pd
import shutil
import mne
import json
import jsonlines
from multiprocessing import Pool
# 定义输入和输出文件路径
def ungz(input_file):
    output_file = input_file[:-3]
    if os.path.exists(output_file):
        return output_file,'existed'
    # 解压缩文件
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return output_file,'ungz ok'

def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path
def convert_gz_to_wav(input_file, output_file):
    with gzip.open(input_file, 'rb') as f_in:
        npz_data = np.load(f_in)
        # print(npz_data.files)
        audio=npz_data['audio']
        fs=npz_data['fs']
        resampled_audio=librosa.resample(audio,orig_sr=fs,target_sr=target_sr)
        sf.write(makedirs(output_file),resampled_audio,samplerate=target_sr)

def read_bdf_gz(input_file):
    output_file,_=ungz(input_file)
    raw=mne.io.read_raw_bdf(output_file,preload=True,verbose=False)
    return raw


def preprocess_raw(raw:mne.io.Raw):
    # 这个只做选channel和滤波操作
    raw.pick(picks=raw.ch_names[:64])
    raw.notch_filter(50, verbose=False)
    raw.filter(l_freq=1, h_freq=60, verbose=False)
    raw.resample(target_eeg_sr)
    return raw.get_data()


def find_files_with_extension(folder_path, extension):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(file_path)
    return file_paths


def get_all_events_tsv(root_dir):
    events_tsvs = find_files_with_extension(root_dir, '.tsv')
    events_tsvs = [i for i in events_tsvs if
                   ('lbollens/sparrkulee/sub-' in i and 'events.tsv' in i and 'listeningActive' in i)]
    return events_tsvs


def read_tsv(tsv_path):
    tsv=pd.read_csv(tsv_path,delimiter='\t')
    tsv_dict={k:tsv[k][0] for k in tsv.keys()}
    return tsv_dict


def read_transcription(path):
    with open(path, 'r', encoding='utf-8') as f:
        transcription=json.load(f)
        # print(transcription)
        return transcription

def check_wav(path):
    try:
        wav,sr=sf.read(path)
        return True
    except Exception as e:
        return False

def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)

def get_data_from_tsv(tsv_path):
    meg_gz_path=tsv_path.replace('events.tsv','eeg.bdf.gz')
    output_data_dir=tsv_path.replace(subjects_dir,output_dir).split('_events')[0]
    assert meg_gz_path!=tsv_path
    assert os.path.dirname(output_data_dir)!=os.path.dirname(tsv_path)
    os.makedirs(output_data_dir,exist_ok=True)
    # subject_id=int(os.path.basename(tsv_path).split('_')[0].split('-')[1])
    raw=read_bdf_gz(meg_gz_path)
    data=preprocess_raw(raw)
    tsv=read_tsv(tsv_path)
    onset_sample=int(tsv['onset']*target_eeg_sr)
    offset_sample=int(tsv['duration']*target_eeg_sr+onset_sample)
    data=data[:,onset_sample:offset_sample]
    stim_file_name=tsv['stim_file'].split('/')[1].split('.')[0]
    wav_path=os.path.join(wav_dir,stim_file_name+'.wav')
    wav,sr=sf.read(wav_path)
    assert len(wav.shape)==1
    transcription_path=os.path.join(transcription_dir,stim_file_name+'.json')
    transcription=read_transcription(transcription_path)
    segments=transcription['segments']
    # 开始切割
    lines=[]
    for i,segment in enumerate(segments):
        segment_duration=segment['end']-segment['start']
        eeg_start=int(segment['start']*target_eeg_sr)
        eeg_end=int(segment['end']*target_eeg_sr)
        text=segment['text']
        if text=='':
            continue
        sliced_eeg=data[:,eeg_start:eeg_end]
        output_wav_path=os.path.join(output_dir,'sliced_audio',stim_file_name,f'{i}.wav')
        if not check_wav(output_wav_path):
        # if 1:
            audio_start=int(segment['start']*target_sr)
            audio_end=int(segment['end']*target_sr)
            sliced_audio=wav[audio_start:audio_end]
            makedirs(output_wav_path)
            sf.write(output_wav_path,sliced_audio,samplerate=target_sr)
        # 保存eeg
        eeg_path=os.path.join(output_data_dir,f'eeg_seg{i}.npy')
        np.save(eeg_path,sliced_eeg)
        line = {
            "speech": {"path": output_wav_path, 'sr': target_sr},
            "eeg": {"path": eeg_path, 'sr': target_eeg_sr},
            "duration": segment_duration,
            "sentence": text.strip(),
            "sentences": [{"text": text.strip(),
                           "start": 0.0, "end": segment_duration, "duration": segment_duration,
                           "words":[{'word':word['word'].strip(),'start':word['start']-segment['words'][0]['start'],'end':word['end']-segment['words'][0]['start']} for word_i,word in enumerate(segment['words'])]
                           }],
            'subj': int(os.path.basename(tsv_path)[4:7]),
            'language':"Dutch"
        }
        lines.append(line)
    jsonl_path=os.path.join(output_data_dir,f'data.jsonl')
    write_jsonlines(jsonl_path,lines)
    return lines


def process(tsv_path):
    lines=[]
    try:
        lines=get_data_from_tsv(tsv_path)
    except Exception as e:
        print(e)
    return lines


if __name__=='__main__':
    target_sr = 16000
    target_eeg_sr = 200
    wav_dir='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/sparrkulee/stimuli/processed_audio/speech'
    transcription_dir='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/sparrkulee/stimuli/processed_audio/transcribe'
    output_dir='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1'
    subjects_dir='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/sparrkulee'
    events_tsvs=get_all_events_tsv(subjects_dir)

    # results=get_data_from_tsv(events_tsvs[0])
    pool = Pool(processes=32)
    results = pool.map(process, events_tsvs)
    pool.close()
    pool.join()
    # print(results[0])
    all_lines1 = []
    for lines in results:
        all_lines1.extend(lines)
    # print(all_lines1)
    write_jsonlines(
        os.path.join(output_dir,'all_info.jsonl'),results)

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