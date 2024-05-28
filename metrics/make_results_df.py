import numpy as np
import pandas as pd
import json

def make_df_results(list, exp_name):
    '''
    input contain json file path list
    '''
    metrics = ['nltkbleu_sentence-1','bert_p', 'bert_r','bert_f','rougeL_fmeasure', 'rouge1_fmeasure', 'wer','cer'] # nltkbleu-1 , 'sacre bleu-2', 'nltkbleu-1', 'rouge1_fmeasure', 'wer', 'cer'
    final_df = []
    for f in list:
        data = []
        with open(f, 'r') as file:
            for line in file:
                # 각 줄을 JSON으로 파싱하고 리스트에 추가합니다.
                json_object = json.loads(line.strip())
                data.append(json_object)
        # print(data[0][''])
        data_filtered = {metric: [data[0].get(metric, None)] for metric in metrics}
        filtered_df_columns = pd.DataFrame(data_filtered)
        final_df.append(filtered_df_columns)
    final_df_set = pd.concat(final_df)
    final_df_set.index = exp_name
    return final_df_set
