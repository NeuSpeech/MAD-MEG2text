import mne
import datetime
import json

sub_ses_dict={}
subjs=list(range(1,28))
for i in subjs:
    try:
        subj='{:02d}'.format(i)
        raw_path1=f'/hpc2hdd/home/yyang937/datasets/gwilliams2023/download/sub-{subj}/ses-0/meg/sub-{subj}_ses-0_task-1_meg.con'
        raw_path2=f'/hpc2hdd/home/yyang937/datasets/gwilliams2023/download/sub-{subj}/ses-1/meg/sub-{subj}_ses-1_task-1_meg.con'

        meg = mne.io.read_raw_kit(raw_path1, preload=False, verbose=False)
        a=meg.info['meas_date']
        meg = mne.io.read_raw_kit(raw_path2, preload=False, verbose=False)
        b=meg.info['meas_date']
        a = datetime.datetime(a.year, a.month, a.day)
        b = datetime.datetime(b.year, b.month, b.day)
        delta = b - a
        days = delta.days
        sub_ses_dict[i]=days
        del a,b,days,meg,delta
    except Exception as e:
        pass
meta_output_path='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/interval_days.json'
with open(meta_output_path, "w") as json_file:
    json.dump(sub_ses_dict, json_file, indent=4,ensure_ascii=False)