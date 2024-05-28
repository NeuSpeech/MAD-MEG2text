import torch
import numpy as np
import copy


class EEGTextAug:
    def __init__(self,
                 max_signal_duration=30, sampling_rate=200,
                 add_sec=0.5,
                 # least time shift is 0, unit is seconds
                 min_time_shift=0, max_time_shift=5,

                 ):
        self.add_sec = add_sec
        self.max_signal_duration = max_signal_duration
        self.sampling_rate = sampling_rate
        self.min_time_shift = min_time_shift
        self.max_time_shift = max_time_shift
        self.max_signal_length = int(sampling_rate * max_signal_duration)
        self.funcs = [
            'time_shift',
            'words_select', 'words_mask',
            'single_words_select', 'single_words_mask',
        ]

    def time_shift(self, unit):
        """
        shift the signal and transcripts.
        sometimes we don't need word level timing,
        but we still provide that just in case.
        """
        transcript = unit['sentences']
        sample = unit['eeg_raw']
        length = max(int((transcript[-1]["end"]) * self.sampling_rate),
                     sample.shape[1])
        assert length < self.max_signal_length, 'signal should be shorter than max signal length'

        max_allow_time_shift=min(self.max_signal_duration-length/self.sampling_rate,self.max_time_shift)
        time_shift=np.random.uniform(low=self.min_time_shift,high=max_allow_time_shift)
        index_shift=int(time_shift*self.sampling_rate)
        # ensure the end point not hitting the boundary
        index_shift = np.clip(index_shift,a_min=0,a_max=self.max_signal_length-length)
        # shift the signal by inserting blank before the signal
        sample = np.pad(sample, [[0, 0], [index_shift, 0]])
        # update time in script
        for sentence in transcript:
            sentence['start']+=time_shift
            sentence['end']+=time_shift
            sentence['words']=[{'word':word['word'],
                                'start':word['start']+time_shift,
                                'end':word['end']+time_shift,
                                } for word in sentence['words']]
        unit['sentences']=transcript
        unit['eeg_raw']=sample
        return unit

    def words_select(self, unit,chosen_num=None,random_shift=10):
        """
        random select a sequence of words from words in sentence and get the segment from the signal.
        This must be processed first. Otherwise, the word time might be incorrect.
        Words time is not accurate after selecting too, because there is some random inserting.
        """
        unit=copy.deepcopy(unit)
        add_sec=np.random.uniform(low=0,high=self.add_sec)
        words, eeg, sr=unit['sentences'][0]['words'],unit['eeg_raw'],self.sampling_rate
        # 修正words时间
        words = [
            {"word": word['word'], "start": word['start'] - words[0]['start'], "end": word['end'] - words[0]['start']}
            for word in words]

        selected_idx = np.random.choice(np.arange(len(words)))
        if chosen_num is None:
            chosen_num = np.random.randint(len(words) - selected_idx)
        assert len(words) != 0
        words = words[selected_idx:selected_idx + chosen_num + 1]
        eeg_len = eeg.shape[1]
        start_index = int(sr * words[0]['start']) + np.random.randint(-random_shift, random_shift)
        end_index = int(sr * (words[-1]['end'] + add_sec)) + np.random.randint(-random_shift, random_shift)
        start_index = np.clip(start_index, 0, eeg_len)
        end_index = np.clip(end_index, 0, eeg_len)
        eeg=eeg[:, start_index:end_index]
        eeg_time=eeg.shape[1]/self.sampling_rate
        words = [
            {"word": word['word'], "start": word['start'] - words[0]['start'], "end": word['end'] - words[0]['start']}
            for word in words]
        # compose words into sentence
        if 'language' not in unit.keys() or unit['language'] in ['English','Dutch']:
            unit['sentence']=' '.join([word['word'] for word in words])
        elif unit['language'] in ['Chinese']:
            unit['sentence']=''.join([word['word'] for word in words])
        else:
            raise NotImplementedError
        unit['sentences'][0]['text']=unit['sentence']
        unit['sentences'][0]['words']=words
        unit['sentences'][0]['start']=0
        unit['sentences'][0]['end']=eeg_time
        unit['start']=0
        unit['end']=eeg_time
        unit['duration']=eeg_time
        return unit

    def words_mask(self, unit):
        """

        """
        raise NotImplementedError

    def __call__(self, unit, func):
        if func == 'time_shift':
            unit=self.time_shift(unit)
        elif func == 'words_select':
            unit=self.words_select(unit)
        else:
            raise NotImplementedError
        return unit
