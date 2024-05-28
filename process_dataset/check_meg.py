# 这个要看每个单词前后是否出现了反应。
# 将单词出现-0.1s~0.6s的数据，把所有的数据在每个通道上做平均，画出来。
# 如果正常数据，会看到0.1s附近有突起。


import jsonlines
import numpy as np
import matplotlib.pyplot as plt


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts


file_path='/data/johj/MEG/gwilliams2023/preprocess7/split3/the_black_willow/lw1/test.jsonl'
all_eeg_data=[]
jsl=read_jsonlines(file_path=file_path)
for js in jsl:
    eeg_data=np.load(js["eeg"]["path"])
    eeg_data_length=eeg_data.shape[1]
    eeg_sr=js["eeg"]['sr']
    words=js["sentences"][0]["words"]
    start_sec=js['start']-js['audio_start']
    for word in words:
        word_start_eeg=word['start']-start_sec
        # 先不管边缘上的单词
        wanted_start=word_start_eeg-0.1
        wanted_start_idx=int(wanted_start*eeg_sr)
        wanted_end_idx=wanted_start_idx+int(0.7*eeg_sr)
        if wanted_start_idx<0 or wanted_end_idx>=eeg_data_length:
            continue
        word_eeg=eeg_data[:,wanted_start_idx:wanted_end_idx]
        assert word_eeg.shape==(208,140)
        all_eeg_data.append(word_eeg)
all_eeg_data=np.array(all_eeg_data)
all_eeg_data=all_eeg_data.mean(axis=0)
all_eeg_data1=all_eeg_data
plt.figure()
for i in range(all_eeg_data1.shape[0]):
    all_eeg_data1[i]=all_eeg_data1[i]-all_eeg_data1[i].mean()
    plt.plot(np.arange(140)/200-0.1,all_eeg_data1[i])
plt.xticks(np.arange(7)*0.1-0.1)
plt.show()
