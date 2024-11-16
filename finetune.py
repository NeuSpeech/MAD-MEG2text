import argparse
import functools
import os
import warnings
import torch
# import torch._dynamo as dynamo
from peft import get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor, AutoConfig, WhisperForConditionalGeneration
from utils.callback import SavePeftModelCallback1
from utils.data_utils import DataCollatorBrainSpeechSeq2SeqWithPadding,get_part_of_dataset
from utils.model_utils import load_from_checkpoint
from utils.load_model import BrainWhisperForConditionalGeneration2, match_modules_string
from utils.reader import BetterDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments
from wandb_callback import WandbPredictionProgressCallback
import datetime

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="/home/yyang/dataset/multi_media/formal_dataset/cut/train_data.jsonl",       help="Path to training data set")
add_arg("test_data",     type=str, default="/home/yyang/dataset/multi_media/formal_dataset/cut/val_data.jsonl",        help="Path to test data set")
add_arg("base_model",    type=str, default="/home/yyang/dataset/multi_media/transformers_whisper_models/large_finetune",      help="Whisper base model")
add_arg("lora_model",    type=str, default=None,      help="trained lora model")
add_arg("mmd_input_type",    type=str, default='mean',      help="mmd")
add_arg("warmup_steps",  type=int, default=10000,      help="warmup_steps")
add_arg("logging_steps", type=int, default=100,     help="logging_steps")
add_arg("eval_steps",    type=int, default=1000,    help="eval_steps")
add_arg("save_steps",    type=int, default=1000,    help="save_steps")
add_arg("num_workers",   type=int, default=6,       help="num_workers")
add_arg("learning_rate", type=float, default=1e-3,  help="learning_rate")
add_arg("modal", type=str, default='speech',  help="modal")
add_arg("sampling_rate", type=int, default=200,  help="sampling_rate")
add_arg("orig_sample_rate", type=int, default=200,  help="orig_sample_rate")
add_arg("eeg_ch", type=int, default=224,  help="eeg_ch")
add_arg("lora_eeg_ch", type=int, default=None,  help="lora_eeg_ch")
add_arg("min_audio_len", type=float, default=0.5,   help="min_audio_len")
add_arg("max_audio_len", type=float, default=30,    help="max_audio_len")
add_arg("use_adalora",   type=bool,  default=True,  help="use_adalora")
add_arg("fp16",          type=bool,  default=False,  help="using fp16")
add_arg("use_8bit",      type=bool,  default=False, help="use_8bit")
add_arg("filter_dataset",      type=bool,  default=False, help="filter_dataset")
add_arg("timestamps",    type=bool,  default=True, help="timestamps")
add_arg("local_files_only", type=bool, default=True, help="local_files_only")
add_arg("num_train_epochs", type=int, default=30,      help="num_train_epochs")
add_arg("language",      type=str, default="English", help="Set the language, which can be the full name or abbreviation. If it is None, the training is multi-language.")
add_arg("task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="Model tasks")
add_arg("augment_config_path",         type=str, default='configs/augmentation.json', help="Data augmentation configuration file path")
add_arg("resume_from_checkpoint",      type=str, default=None, help="Checkpoint path to resume training")
add_arg("per_device_train_batch_size", type=int, default=2,    help="train batch size")
add_arg("per_device_eval_batch_size",  type=int, default=2,    help="eval batch size")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="Gradient accumulation steps")
add_arg("fine_tune_layers", type=int, default=None,    help="Fine-tune the first few layers of the base model")
add_arg("device", type=str, default='auto',    help="device")
add_arg("config_name", type=str, default='base',    help="conv1 module")
add_arg("data_ratio", type=float, default=None,    help="The proportion of data used in the training set")
add_arg("random_initialize_whisper", type=bool, default=False,    help="Random initialization of whisper")

add_arg("clip", type=float, default=0,    help="loss combination")
add_arg("mse", type=float, default=0,    help="loss combination")
add_arg("mmd_bm", type=float, default=0,    help="loss combination")
add_arg("mmd", type=float, default=0,    help="loss combination")
add_arg("ce", type=float, default=0,    help="loss combination")

add_arg("ft_full", type=bool, default=False, help="Fine-tune the entire model")
add_arg("depth", type=int, default=5,    help="depth in brain_module")
add_arg("trainable_brainmodule", type=bool, default=True, help="Tranable brainmodule encoder")
add_arg("additional_runname", type=str, default='default', help="additional runname in wandb")
args = parser.parse_args()
print_arguments(args)
import os
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
os.environ["WANDB_PROJECT"]="MAD-MEG2TEXT"

loss = ''
loss_dict = {
    'clip': args.clip,
    'mse': args.mse,
    'mmd_bm': args.mmd_bm,
    'mmd': args.mmd,
    'ce': args.ce
}
for key, value in loss_dict.items():
    if value != 0:
        loss += f'{key}_{value}'

run_name = args.additional_runname+'_'+timestamp+'_'+loss+'_'+f'{args.learning_rate}'+'_'+f'{args.per_device_train_batch_size}'+f'_trainable_brainmodule_{args.trainable_brainmodule}_adalora_{args.use_adalora}'
os.environ["WANDB_RUN_ID"]= run_name
output_dir=f'output_model/{run_name}'
# Get Whisper's data processor, which includes feature extractor and tokenizer
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)

# Read data
train_dataset = BetterDataset(
    data_list_path=args.train_data,
    processor=processor,
    modal=args.modal,
    modal_ch=args.eeg_ch,
    sample_rate=args.sampling_rate,
    language=args.language,
    timestamps=args.timestamps,
    min_duration=args.min_audio_len,
    max_duration=args.max_audio_len,
    augment_config_path=args.augment_config_path)
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

if args.data_ratio is not None:
    train_dataset.data_list=get_part_of_dataset(train_dataset.data_list,args.data_ratio)

print(f"train: {len(train_dataset)}, test： {len(test_dataset)}")
# Data padding
data_collator = DataCollatorBrainSpeechSeq2SeqWithPadding(processor=processor)

# Get Whisper model
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
# print(f'device_map:{device_map}, os env:{os.environ["CUDA_VISIBLE_DEVICES"]}')
# device_map = 'cpu'
# device mapping
print(f'device map :{device_map}')

pretrained = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
# whisper_config = args.base_model + '/config.json'
config = AutoConfig.from_pretrained("openai/whisper-base")
depth = args.depth
model = BrainWhisperForConditionalGeneration2(config, loss_dict, pretrained, run_name, depth) 
model.config.mmd_input_type=args.mmd_input_type
device = torch.device(device_map)
model.to(device)
# model.set_brain_module(args.eeg_ch)
# print(f'model device {model.device}')

if args.lora_model is not None:
    # The previous method of loading the model was to change the model into the shape of the model to be loaded, and then load the parameters.
    # Now it becomes the model to be trained.
    model = PeftModel.from_pretrained(model, args.lora_model, local_files_only=args.local_files_only)
    model = model.merge_and_unload()

# model.save_pretrained(save_directory=os.path.join(args.output_dir, "checkpoint-init"))
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
# Quantitative model
model = prepare_model_for_kbit_training(model)
# Register forward, otherwise multi-card training will fail.
model.brain_module.initial_linear.register_forward_hook(make_inputs_require_grad)

for param in model.parameters():
    param.requires_grad=False

if args.use_adalora:
    print(f'adding LoRA modules...')
    # prefixes = [f'model.encoder.layers.{i}.' for i in [0,1,2,3]]
    if args.fine_tune_layers is not None:
        prefixes = [f'model.model.encoder.layers.{i}.' for i in range(args.fine_tune_layers)]
    elif args.ft_full:
        prefixes = ['model.model']
    else:
        prefixes = ['model.model.encoder']
    suffixes = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
    # model_named_modules=[]
    # target_modules = []
    target_modules = match_modules_string(model.named_modules(), prefixes, suffixes)
    print('target_modules')
    print(target_modules)
    modules_to_save= ['brain_module.'+name for name, _ in model.named_modules()][1:]
    # match_modules(model.named_modules(),[''],[''],[".*model.encoder.conv1",".*model.encoder.conv2"])
    # modules_to_save = ['model.encoder.conv1', 'model.encoder.conv2']
    #print('modules_to_save')
    #print(modules_to_save)
    config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                            lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules,
                            modules_to_save=modules_to_save)
    model = get_peft_model(model, config)

if args.trainable_brainmodule:
    for param in model.brain_module.parameters():
        param.requires_grad=True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {trainable_params}')


if args.base_model.endswith("/"):
    args.base_model = args.base_model[:-1]
output_dir = os.path.join(output_dir, os.path.basename(args.base_model))
# Define training parameters
training_args = \
    Seq2SeqTrainingArguments(output_dir=output_dir,  # Directory to save checkpoints and will
                             per_device_train_batch_size=args.per_device_train_batch_size,  # Training batch_size size
                             per_device_eval_batch_size=args.per_device_eval_batch_size,  # Evaluation batch size
                             gradient_accumulation_steps=args.gradient_accumulation_steps,  # Cumulative number of training gradient steps
                             learning_rate=args.learning_rate,  # learning rate
                             warmup_steps=args.warmup_steps,  # Number of warm-up steps
                             num_train_epochs=args.num_train_epochs,  # Fine-tuning the number of training rounds
                             save_strategy="steps",  # Specify the number of steps to save checkpoints
                             evaluation_strategy="steps",  # Specifies the number of steps to evaluate the model
                             load_best_model_at_end=False,  # Specify whether to load the optimal model at the end
                             fp16=args.fp16,  # Whether to use half-precision training
                             report_to=None,  # report to wandb
                             save_steps=args.save_steps,  # Specify the number of steps to save checkpoints
                             eval_steps=args.eval_steps,  # Specify the number of steps to evaluate the model
                             save_total_limit=5,  # Only the latest checkpoint count is saved
                             optim='adamw_torch',  # optimizer
                             max_grad_norm=0.3,
                             ddp_find_unused_parameters=False if ddp else None,  # Distributed training setup
                             dataloader_num_workers=args.num_workers,  # Set the number of threads to read data
                             logging_steps=args.logging_steps,  # Specify the number of steps to print the log
                             remove_unused_columns=False,  # Delete data columns that are not required by the model
                             label_names=["labels"],
                             save_safetensors=False
                             )  # A list of keys in the input dictionary corresponding to the label

# if training_args.local_rank == 0 or training_args.local_rank == -1:
print('trainable parameters')
print('=' * 90)

for name,param in model.named_parameters():
    if param.requires_grad:
        print(name)
print('=' * 90)

# Compiler using Pytorch2.0
# if torch.__version__ >= "2" and platform.system().lower() != 'windows':
#     model = torch.compile(model)

# Define trainer
trainer = Seq2SeqTrainer(args=training_args,
                         model=model,
                         train_dataset=train_dataset,
                         eval_dataset=test_dataset,
                         data_collator=data_collator,
                         tokenizer=processor.feature_extractor,
                         callbacks=[SavePeftModelCallback1],
                         )
# wandb callback
# print(test_dataset)
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=processor.tokenizer,
    val_dataset=test_dataset,
    train_dataset=train_dataset,
    num_samples=10,
)
trainer.add_callback(progress_callback)
print('trainer_callback_list:', trainer.callback_handler.callbacks)

for name, param in model.named_parameters():
    if torch.all(param == 0):
        print(name)
model.config.use_cache = False
trainer._load_from_checkpoint = load_from_checkpoint
resume_from_checkpoint=args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
trainer.save_model(os.path.join(output_dir, "checkpoint-final"))
