import argparse
import functools
import gc
import json
import os
import torch.nn as nn
import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoConfig
from utils.load_model import BrainWhisperForConditionalGeneration2
from utils.model_utils import projection_module
from peft import PeftModel, AdaLoraConfig, get_peft_model
from utils.data_utils import DataCollatorBrainSpeechSeq2SeqWithPadding,generate_random_string, remove_punctuation, to_simple,contains_valid_letters
from utils.process_str import filter_ascii_text, model_generate, convert_lower_text, list_operation
from utils.reader import BetterDataset,write_jsonlines,read_jsonlines
from utils.utils import print_arguments, add_arguments
from utils.generation_helper import GetSequenceBias
import pickle
import re

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="/data/johj/MEG/gwilliams2023/preprocess7/split3/cable_spool_fort/lw1/train.jsonl", help="test set")
add_arg("checkpoint_path",  type=str, default="models/whisper-tiny-finetune", help="full model checkpoint path")
add_arg("model_path",    type=str, default="/data/johj/MEG/transformer_whisper_models", help="whisper")
add_arg("modal", type=str, default='speech',  help="输入的模态")
add_arg("sampling_rate", type=int, default=120,  help="输入信号采样率")
add_arg("eeg_ch", type=int, default=208,  help="输入信号通道数")
add_arg("batch_size",  type=int, default=16,        help="评估的batch size")
add_arg("num_workers", type=int, default=8,         help="读取数据的线程数量")
add_arg("language",    type=str, default="Chinese", help="设置语言，可全称也可简写，如果为None则评估的是多语言")
add_arg("remove_pun",  type=bool, default=True,     help="是否移除标点符号")
add_arg("to_simple",   type=bool, default=True,     help="是否转为简体中文")
add_arg("timestamps",  type=bool, default=False,    help="评估时是否使用时间戳数据")
add_arg("min_audio_len",     type=float, default=0.5,  help="最小的音频长度，单位秒")
add_arg("max_audio_len",     type=float, default=30,   help="最大的音频长度，单位秒")
add_arg("local_files_only",  type=bool,  default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("noise",  type=bool,  default=False, help="输入模型的是噪声")
add_arg("filter_dataset",      type=bool,  default=False, help="是否过滤数据集")
add_arg("random_choice",  type=bool,  default=False, help="随机选择标签中的文本,选用这个，等于模型无效，noise无效")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("random_initialize_whisper", type=bool, default=False,    help="随机初始化whisper")
add_arg("teacher_forcing", type=bool, default=False,    help="使用teacher forcing")
add_arg("extra_name", type=str, default=None,    help="result basename里面增加字符")
add_arg("post_processing", type=bool, default=False,    help="是否使用后处理")
add_arg("config_name", type=str, default='base',    help="使用的模型")
add_arg("add_sequence_bias", type=bool, default=False,    help="是否对生成词增强。")
add_arg("base_model",    type=str, default="/data/johj/MEG/transformer_whisper_models", help="Whisper的基础模型")
add_arg("device", type=str, default='cuda',    help="device")
add_arg("mmd_input_type",    type=str, default='mean',      help="mmd")


# add_arg("metric",     type=str, default="fulleval",        choices=['cer', 'wer','fulleval'],              help="评估方式")
args = parser.parse_args()
print_arguments(args)

# model path checking
assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"The model file {args.model_path} does not exist. Please check whether the model has been successfully merged, or if it is a model available on Hugging Face."
# Get Whisper's data processor, which includes feature extractor and tokenizer
print('loading')


os.environ['WORLD_SIZE'] = '1'
device_map = args.device
if device_map != 'cpu':
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if world_size != 1:
        device_index = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"{device_map}:{device_index}")
    else:
        device = torch.device(f"{device_map}:0")
else:
    device = torch.device("cpu")

''' base model load '''
pretrained = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
whisper_config = args.base_model + '/config.json'
whisper_config = AutoConfig.from_pretrained(whisper_config)
checkpoint_path =args.checkpoint_path
state_dict = torch.load(checkpoint_path+'full_model.pth')
depth=5
model = BrainWhisperForConditionalGeneration2(whisper_config, state_dict.config.total_loss, pretrained, state_dict.config.run_name, depth=depth)
model.config.mmd_input_type=args.mmd_input_type
#model = get_peft_model(model, state_dict.peft_config['default'])

''' merge lora '''
contains_lora = any('lora' in key for key in state_dict.state_dict())
if contains_lora:
    print("adaLora was used")
    model = PeftModel.from_pretrained(model, checkpoint_path, local_files_only=args.local_files_only)
    model = model.merge_and_unload()

''' brain model load '''
brain_module_state_dict = {name.replace('base_model.model.', ''): param for name, param in state_dict.state_dict().items() if 'brain_module' in name}
model.load_state_dict(brain_module_state_dict, strict=False)
print(model)

device = torch.device(device_map)
model.to(device)
if args.noise==True:
    print(args.noise)
    results_path = '/data/johj/results/noise/'+state_dict.config.run_name
    mel_path = '/data/johj/predmel_output/noise/'+state_dict.config.run_name

else:
    results_path = '/data/johj/results/'+state_dict.config.run_name
    mel_path = '/data/johj/predmel_output/'+state_dict.config.run_name

if not os.path.exists(results_path):
    os.makedirs(results_path)

if not os.path.exists(mel_path):
    os.makedirs(mel_path)
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)
print('loading done')
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=args.language,
    task=args.task,
    no_timestamps=not args.timestamps,)


model.eval()

# Get test data

test_dataset = BetterDataset(
    data_list_path=args.test_data,
    processor=processor,
    modal=args.modal,
    modal_ch=args.eeg_ch,
    sample_rate=args.sampling_rate,
    language=args.language,
    timestamps=args.timestamps,
    min_duration=args.min_audio_len,
    max_duration=args.max_audio_len)
print(f"test set size：{len(test_dataset)}")

# Data padding
data_collator = DataCollatorBrainSpeechSeq2SeqWithPadding(processor=processor)

eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator)

# Get Whisper model
metrics = []
# metric_files = ['bert_score','bleu','mer', 'my_rouge','perplexity', 'wer','word_info_lost','word_info_preserved']
# metric_files = ['bleu', 'nltkbleu_sentence', 'sacrebleu', 'mer', 'my_rouge','wer','word_info_lost','word_info_preserved',
#                 'bert_score','meteor','cer1'
#                 ]
metric_files = ['nltkbleu_sentence', 'my_rouge','word_info_preserved',
                'bert_score','cer1'
                ]
# Load metrics
for metric_file in metric_files:
    metric = evaluate.load(f'metrics/{metric_file}.py',
                           experiment_id=generate_random_string(100))
    metrics.append(metric)


if args.random_choice:
    all_labels=[]
result_basename=(f'formal_test_results{"_"+args.extra_name if args.extra_name is not None else ""}'
                 f'{"no_post_processing" if not args.post_processing else "post_processing"}'
                 f'{"_noise"if args.noise else ""}{"_randomChoice"if args.random_choice else ""}'
                 f'{"_tf" if args.teacher_forcing else ""}')
# Evaluation path
output_file=os.path.join(results_path,f'{result_basename}.txt')

if args.add_sequence_bias:
    generation_helper=GetSequenceBias(tokenizer_name=args.model_path,
                                      jsonl_path=args.test_data.replace('test.jsonl','train.jsonl'),
                                      bias=-1.0,
                                      extract_type='phrase_word')
result_preds=[]
result_labels=[]
target_tokens_list=[] # for nltk
pred_tokens_list=[]

def preprocess_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text).strip()
    return text

with open(output_file, "w") as f:
    for step, batch in tqdm.tqdm(enumerate(eval_dataloader)):
        if step==0:
            select_results = {'meg': None, 'decoded_labels': [], 'decoded_preds': [], 'pred_mel': None, 'mel_spec':None}
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                if not args.random_choice:
                    input_features = batch["input_features"].cuda()
                    if args.noise:
                        input_features=torch.randn_like(input_features)
                        print('noise')
                    if not args.teacher_forcing:
                        if args.language.lower() != 'english':
                            decoder_input_ids = batch["labels"][:, :4].cuda()
                            generation_kwargs={"decoder_input_ids":decoder_input_ids}
                        else:
                            generation_kwargs={}
                        if args.add_sequence_bias==True:
                            sequence_bias=generation_helper.get_bias_for_my_sentences()
                            sequence_bias_kwargs={"sequence_bias":sequence_bias}
                            # print(sequence_bias_kwargs)
                        else:
                            sequence_bias_kwargs={}
                        # print(sequence_bias_kwargs,type(args.add_sequence_bias),args.add_sequence_bias==True,args.post_processing)
                        generated_tokens = (
                            model.generate(input_features,
                                           useful_length=400,
                                           subject_index=batch['subject_index'],
                                           mel_spec=batch['mel_spec'],
                                           do_sample=False, num_beams=5,
                                           # do_sample=False,num_beams=20,
                                           # do_sample=True,num_beams=20,typical_p=0.25,
                                           # do_sample=True,num_beams=20,top_p=0.25,
                                           # do_sample=True,num_beams=20,top_p=0.25,
                                           # do_sample=False,num_beams=5,num_beam_groups=5, # 这个能比随机搞2个点
                                           # diversity_penalty=1.0,

                                           repetition_penalty=5.0,no_repeat_ngram_size=2,max_new_tokens=50,
                                           **sequence_bias_kwargs,

                                           **generation_kwargs
                                           # max_new_tokens=100,
                                           # forced_decoder_ids=forced_decoder_ids
                                           )
                        ).cpu().numpy()
                    else:
                        # print(batch["labels"])
                        # print(batch["labels"].shape)
                        # exit()
                        # 50257
                        indices=batch["labels"]==-100
                        batch["labels"][indices]=50257
                        # decoder_input_ids=batch["labels"].cuda()
                        model_output=model(input_features, subject_index=batch['subject_index'].cpu().cuda(), useful_length=400, mel_spec=batch['mel_spec'].cuda(), labels=batch["labels"].cuda())
                        logits=model_output.logits
                        # logits=logits.to('cpu').numpy()
                        # print(f'logits shape:{logits.shape}')
                        values,predictions=logits.softmax(dim=-1).topk(1)
                        # print(f'predictions shape:{predictions.shape}')
                        predictions=torch.squeeze(predictions,dim=-1)
                        # print(f'predictions:{predictions}')
                        generated_tokens=predictions.cpu().numpy()
                        generated_tokens[indices]=-100
                        # print(f'generated_tokens:{generated_tokens.shape}')
                        


                labels = batch["labels"].cpu().numpy()
                # print(f'labels:{labels}')
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                
                for label in labels:
                    target_tokens_list += [list(label)]
                for tok in generated_tokens:
                    pred_tokens_list += [list(tok)]
                # 将预测和实际的 token 转换为文本
                if not args.random_choice:
                    decoded_preds = processor.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
                result_preds.extend(decoded_preds)
                result_labels.extend(decoded_labels)
                # decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                # decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                if args.post_processing:
                    decoded_preds=filter_ascii_text(decoded_preds)
                    decoded_labels=filter_ascii_text(decoded_labels)
                    decoded_preds=convert_lower_text(decoded_preds)
                    decoded_labels=convert_lower_text(decoded_labels)
                    decoded_preds=[preprocess_text(pred) for pred in decoded_preds]

                for pred, label in zip(decoded_preds, decoded_labels):
                    f.write(f"start********************************\n")
                    f.write(f"Predicted: {pred}\n")
                    f.write(f"True: {label}\n")
                    f.write(f"end==================================\n\n")

                if not args.random_choice:
                    print('decoded_preds')
                    print(decoded_preds)
                    print('decoded_labels')
                    print(decoded_labels)
                    print('\n')
                    print('end')
                if args.random_choice:
                    all_labels.extend(decoded_labels)
                if args.teacher_forcing:
                    if step % 150 == 0:
                        print("start producing mel results********************************")
                        if select_results['meg']==None:
                            select_results['meg'] = input_features.cpu()
                            select_results['decoded_labels']+=decoded_labels
                            select_results['decoded_preds']+=decoded_preds
                            select_results['pred_mel'] = model_output.p_mel.cpu()
                            select_results['mel_spec'] = model_output.mel.cpu()
                        else:
                            select_results['meg'] = torch.cat((select_results['meg'], input_features.cpu()), axis=0) 
                            select_results['decoded_labels']+=decoded_labels
                            select_results['decoded_preds']+=decoded_preds
                            select_results['pred_mel'] = torch.cat((select_results['pred_mel'], model_output.p_mel.cpu()), axis=0)
                            select_results['mel_spec'] = torch.cat((select_results['mel_spec'], model_output.mel.cpu()), axis=0) # .cpu().detach().numpy()

                        print("end mel results********************************")

                

                # torch.cat((torch.randn(3,5),torch.randn(3,5)), axis=0)

# python 3.12.2
with open(f'{mel_path}/select_results.pkl', 'wb') as pkl_file:
    pickle.dump(select_results, pkl_file)

if not args.random_choice:
    jsonl_file_path=os.path.join(results_path,f'{result_basename}.jsonl')
    jsonl_file=[{"pred":pred,"label":label} for pred,label in zip(select_results['decoded_preds'], select_results['decoded_labels'])]
    write_jsonlines(jsonl_file_path, jsonl_file)
    for metric in metrics:
        # print(metric.description)
        if metric.description=='nltk':
            metric.add_batch(pred_tokens_list=pred_tokens_list, target_tokens_list=target_tokens_list)
        else:
            metric.add_batch(predictions=select_results['decoded_preds'], references=select_results['decoded_labels'])

if not args.random_choice:
    results={}
    for metric in metrics:
        result = metric.compute()
        for key in result.keys():
            if type(result[key])==torch.Tensor:
                result[key]=result[key].item()
            results[key]=result[key]
    print(f"Evaluation results：{results}")
    json_file_path=os.path.join(results_path,f'{result_basename}.json')
    with open(json_file_path,'w') as f:
        json.dump(results,f)


# random_choice
if args.random_choice:
    all_preds=np.random.choice(all_labels,len(all_labels))
    for metric in metrics:
        metric.add_batch(predictions=all_preds, references=all_labels)
    results = {}
    for metric in metrics:
        result = metric.compute()
        for key in result.keys():
            if type(result[key]) == torch.Tensor:
                result[key] = result[key].item()
            results[key] = result[key]
    print(f"Evaluation results: {results}")
    json_file_path = os.path.join(results_path, f'{result_basename}.json')
    with open(json_file_path, 'w') as f:
        json.dump(results, f)
