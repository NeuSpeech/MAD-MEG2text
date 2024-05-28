import soundfile as sf
import librosa
import whisper
import json
import numpy as np
import copy
import os
import mne
import librosa
from multiprocessing import Pool
import jsonlines
import tqdm



def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path


import soundfile as sf
import os
import json
import whisper
import tqdm
import librosa

model = whisper.load_model('large')


def makedirs(output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    return output_dir



# 先把语音转录文件都写了。
# 然后在轮subject，取eeg。
def get_speed_transcription(path='musicx1_transcribe rectified.json'):
    path=os.path.join(root_path,path)
    with open(path, 'r', encoding='utf-8') as f:
        transcription = json.load(f)

    for i in range(5):
        speed=i+1
        trans=copy.deepcopy(transcription)
        new_trans={
            "text":''.join([trans['segments'][j]['text'] for j in range(len(trans['segments']))]),
            "segments":[{
                "text":trans['segments'][j]['text'],
                "start":trans['segments'][j]['start']/speed,
                "end":trans['segments'][j]['end']/speed,
            } for j in range(len(trans['segments']))]
        }
        new_trans_path=f'transcribe/{speed}x.json'
        new_trans_path=os.path.join(root_path,new_trans_path)
        makedirs(new_trans_path)
        with open(new_trans_path, "w") as json_file:
            json.dump(new_trans, json_file, indent=4, ensure_ascii=False)


def get_onsets(raw):
    copy_raw=copy.deepcopy(raw)
    copy_raw.pick(picks='stim', verbose=False)
    copy_raw=copy_raw.load_data( verbose=False)
    eeg_sr=copy_raw.info['sfreq']
    data=copy_raw.get_data(verbose=False)
    onsets=[]
    for i in range(-6,-1):
        onsets.append(np.argmax(data[i])/eeg_sr)
    return onsets


def preprocess_raw(raw:mne.io.Raw):
    # 这个只做选channel和滤波操作
    raw.pick(picks=raw.ch_names[:256], verbose=False)
    raw=raw.load_data(verbose=False)
    raw.notch_filter(50, verbose=False)
    raw.filter(l_freq=1, h_freq=49, verbose=False)
    raw.resample(target_eeg_sr, verbose=False)
    return raw.get_data()


def cut_audio():
    transcribe_dir=os.path.join(root_path,'transcribe')
    for i in range(5):
        speed=i+1
        transcribe_path=os.path.join(transcribe_dir,f'{speed}x.json')
        with open(transcribe_path, 'r', encoding='utf-8') as f:
            transcription_dict = json.load(f)
        segments=transcription_dict['segments']
        read_music_path=os.path.join(root_path,f'music x {speed}.mp3')
        wav,wav_sr=sf.read(read_music_path,always_2d=True)
        if wav_sr!=16000:
            wav=wav[:,0]
            wav=librosa.resample(wav,orig_sr=wav_sr,target_sr=target_sr)
        for j,segment in enumerate(segments):
            onset=int(segment['start']*target_sr)
            offset=int(segment['end']*target_sr)
            seg_wav=wav[onset:offset]
            seg_wav_path=os.path.join(root_path,output_data_dir,f'speed_{speed}','music',f'seg_{j}.wav')
            sf.write(makedirs(seg_wav_path),seg_wav,samplerate=target_sr)




def get_data_from_subject(subj):
    subj_path=os.path.join(root_path,f'subj{subj}.mff')
    transcribe_dir=os.path.join(root_path,'transcribe')
    raw = mne.io.read_raw_egi(subj_path, verbose=False, preload=False)
    onsets=get_onsets(raw)
    data=preprocess_raw(raw)
    lines =[]
    for i,onset in enumerate(onsets):
        speed=i+1
        transcribe_path=os.path.join(transcribe_dir,f'{speed}x.json')
        with open(transcribe_path, 'r', encoding='utf-8') as f:
            transcription_dict = json.load(f)
        segments=transcription_dict['segments']

        for j,seg in enumerate(segments):
            start_sec=seg['start']
            end_sec=seg['end']
            eeg_start_sec=onset+start_sec
            eeg_end_sec=onset+end_sec
            eeg_start_index=int(eeg_start_sec*target_eeg_sr)
            eeg_end_index=int(eeg_end_sec*target_eeg_sr)
            eeg_seg=data[:,eeg_start_index:eeg_end_index]
            eeg_seg_path=os.path.join(root_path,output_data_dir,f'speed_{speed}',f'subj_{subj}',f'seg_{j}.npy')
            makedirs(eeg_seg_path)
            np.save(eeg_seg_path,eeg_seg)

            line={
            "speech": {"path": os.path.join(root_path,output_data_dir,f'speed_{speed}',f'seg_{j}.wav'), 'sr': target_sr},
            "eeg": {"path": eeg_seg_path, 'sr': target_eeg_sr},
            "duration": end_sec-start_sec,
            "sentence": seg['text'].strip(),
            "sentences": [{"text": seg['text'].strip(),
                           "start": 0.0, "end": end_sec-start_sec, "duration": end_sec-start_sec
                           }],
            'subj': subj,
            'speed': speed,
            'language':"Chinese"
            }
            lines.append(line)
    return lines


def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)


def get_all_lines():
    subj_list=[i+1 for i in range(20)]
    if use_multiprocessing:
        pool = Pool(processes=20)
        results = pool.map(get_data_from_subject, subj_list)
        pool.close()
        pool.join()
        # print(results[0])
        all_lines1 = []
        for lines in results:
            all_lines1.extend(lines)
    else:
        all_lines1=[]
        for i in tqdm.tqdm(range(20)):
            all_lines1.extend(get_data_from_subject(i+1))
    # print(all_lines1)
    write_jsonlines(
        os.path.join(root_path,output_data_dir, 'all_info.jsonl'), all_lines1)


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts


def select_json_lines(jsonline_path,speed,):
    lines=read_jsonlines(jsonline_path)
    selected_lines=[]
    for line in lines:
        if line['speed']==speed:
            selected_lines.append(line)
    return selected_lines



if __name__ == "__main__":
    root_path='/hpc2hdd/home/yyang937/datasets/music'
    output_data_dir='processed1'
    target_eeg_sr=200
    target_sr=16000
    use_multiprocessing=True

    # cut_audio()
    # get_all_lines()
    for speed in range(1,6):
        lines=select_json_lines(os.path.join(root_path,output_data_dir, 'all_info.jsonl'),speed=speed)

        write_jsonlines(
            os.path.join(root_path,output_data_dir,
                         f'speed_{speed}', f'speed_{speed}_info.jsonl'), lines)
