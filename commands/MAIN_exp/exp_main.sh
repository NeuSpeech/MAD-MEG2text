# train
CUDA_VISIBLE_DEVICES=0 python finetune.py --per_device_train_batch_size=32\
 --per_device_eval_batch_size=32 --mmd_bm=0.1 --mmd=0.1\
 --eval_steps=1000 --save_steps=1000 --trainable_brainmodule=True --use_adalora=False\
 --learning_rate=3e-4 --fp16=True --timestamps=False\
 --num_train_epochs=5 --warmup_steps=5 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=100 --orig_sample_rate=100\
 --train_data='/NFS/Users/johj/gwilliams2023/preprocess8/split3/cable_spool_fort/lw1/train.jsonl'\
 --test_data='/NFS/Users/johj/gwilliams2023/preprocess8/split3/cable_spool_fort/lw1/val.jsonl'\
 --base_model='/data/johj/MEG/transformer_whisper_models' --augment_config_path=None\
 --local_files_only=True --language='English' --device='cuda' --additional_runname='Exp'

# non teacher-forcing
# CUDA_VISIBLE_DEVICES=0 python evaluation.py\
#  --checkpoint_path='../neuspeech/output_model/exp10_0521_0213_clip_1.0_0.0003_32_trainable_brainmodule_True_adalora_False/transformer_whisper_models/checkpoint-6000/'\
#  --model_path='/data/johj/MEG/transformer_whisper_models'\
#  --test_data='/NFS/Users/johj/gwilliams2023/preprocess8/split3/cable_spool_fort/lw1/test.jsonl'\
#  --modal='eeg' --sampling_rate=100 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
#  --timestamps=False --local_files_only=True --teacher_forcing=False --post_processing=True 

# # noise non teacher-forcing
# CUDA_VISIBLE_DEVICES=0 python evaluation.py\
#  --checkpoint_path='output_model/exp10_0521_0213_clip_1.0_0.0003_32_trainable_brainmodule_True_adalora_False/transformer_whisper_models/checkpoint-6000/'\
#  --model_path='/data/johj/MEG/transformer_whisper_models'\
#  --test_data='/NFS/Users/johj/gwilliams2023/preprocess8/split3/cable_spool_fort/lw1/test.jsonl'\
#  --modal='eeg' --sampling_rate=100 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
#  --timestamps=False --local_files_only=True --teacher_forcing=False --post_processing=True --noise=True

# # teacher-forcing
# CUDA_VISIBLE_DEVICES=0 python evaluation.py\
#  --checkpoint_path='output_model/exp10_0521_0213_clip_1.0_0.0003_32_trainable_brainmodule_True_adalora_False/transformer_whisper_models/checkpoint-6000/'\
#  --model_path='/data/johj/MEG/transformer_whisper_models'\
#  --test_data='/NFS/Users/johj/gwilliams2023/preprocess8/split3/cable_spool_fort/lw1/test.jsonl'\
#  --modal='eeg' --sampling_rate=100 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
#  --timestamps=False --local_files_only=True --teacher_forcing=True --post_processing=True 

# # noise teacher-forcing
# CUDA_VISIBLE_DEVICES=0 python evaluation.py\
#  --checkpoint_path='output_model/exp10_0521_0213_clip_1.0_0.0003_32_trainable_brainmodule_True_adalora_False/transformer_whisper_models/checkpoint-6000/'\
#  --model_path='/data/johj/MEG/transformer_whisper_models'\
#  --test_data='/NFS/Users/johj/gwilliams2023/preprocess8/split3/cable_spool_fort/lw1/test.jsonl'\
#  --modal='eeg' --sampling_rate=100 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
#  --timestamps=False --local_files_only=True --teacher_forcing=True --post_processing=True --noise=True