import soundfile as sf
import os
import json
import whisper
import tqdm
import librosa
import sys
# Get the file path of the current script
current_path = os.path.abspath(__file__)
# Get the path to the project root directory
project_root = os.path.dirname(os.path.dirname(current_path))
# Add the project root directory to sys.path
sys.path.append(project_root)

import argparse
import functools
from utils.utils import add_arguments

def makedirs(output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    return output_dir


# python process_dataset/asr_pipeline.py --input_dir="datasets/gwilliams2023/download/stimuli/audio"
# --output_dir="datasets/gwilliams2023/preprocess6/audio"

if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("input_dir",     type=str, default=None,        help="需要设置的存放音频文件的路径")
    add_arg("type",     type=str, default='wav',        help="音频文件的后缀")
    add_arg("output_dir",     type=str, default=None,        help="输出的音频和转写文本的位置")
    add_arg("force_rewrite",     type=bool, default=True,        help="强制重写所有音频文件到新文件夹")
    add_arg("language",     type=str, default=None,        help="设置语言")
    add_arg("backend",     type=str, default='whisperx',        help="使用的模型")
    args = parser.parse_args()

    wav_dir = os.path.join(home_dir,args.input_dir)
    wav16_dir = os.path.join(home_dir,args.output_dir,'wav16')
    transcription_dir = os.path.join(home_dir,args.output_dir,'transcription')
    wav_files = [i for i in os.listdir(wav_dir) if i.split('.')[-1] == args.type]
    print(wav_files)
    assert len(wav_files)!=0,'there is no file under the input dir!'
    target_sr = 16000
    language = args.language
    if args.backend=='whisper':
        model = whisper.load_model('large')
    elif args.backend=='whisperx':
        import whisperx

        batch_size = 16
        device='cuda'
        compute_type='float16'
        model = whisperx.load_model("large-v2", 'cuda', compute_type=compute_type)
    else:
        raise NotImplementedError

    # Transcribe from audio to text. You also need to convert the audio to 16kHz and write it into a json file
    for wav_path in tqdm.tqdm(wav_files):
        wav_path = os.path.join(wav_dir, wav_path)
        wav_name = os.path.basename(wav_path).split('.')[0]
        wav, wav_sr = sf.read(wav_path, always_2d=True)
        wav = wav[:, 0]
        if wav_sr != 16000 or args.force_rewrite:
            wav = librosa.resample(wav, orig_sr=wav_sr, target_sr=target_sr)
            wav16_path = os.path.join(wav16_dir, f'{wav_name}_16kHz.wav')
            sf.write(makedirs(wav16_path), wav, samplerate=target_sr)
            wav_path=wav16_path

        if args.backend == 'whisper':
            result = model.transcribe(
                wav_path, language=language, word_timestamps=True,
                without_timestamps=False)
        elif args.backend=='whisperx':
            audio = whisperx.load_audio(wav_path)
            result = model.transcribe(audio, batch_size=batch_size)
            # 2. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result['language'], device=device)
            segments_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            for k in ['segments','word_segments']:
                result[k]=segments_result[k]
        else:
            raise NotImplementedError

        # write text
        transcribe_path = f"{transcription_dir}/{os.path.basename(wav_path).split('.')[0]}.json"
        transcribe_path = makedirs(transcribe_path)
        with open(transcribe_path, 'w') as write_f:
            json.dump(result, write_f, indent=4, ensure_ascii=False)