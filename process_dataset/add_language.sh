python process_dataset/add_language.py --jsonl "datasets/schoffelen2019n/preprocess6/audio_info.jsonl"\
 "datasets/schoffelen2019n/preprocess6/audio_split1/train.jsonl" \
"datasets/schoffelen2019n/preprocess6/audio_split1/val.jsonl" \
"datasets/schoffelen2019n/preprocess6/audio_split1/test.jsonl" \
--language="Dutch"

python process_dataset/add_language.py --jsonl \
"datasets/gwilliams2023/preprocess5/info.jsonl" \
"datasets/gwilliams2023/preprocess5/split1/train.jsonl" \
"datasets/gwilliams2023/preprocess5/split1/val.jsonl" \
"datasets/gwilliams2023/preprocess5/split1/test.jsonl" \
"datasets/gwilliams2023/preprocess5/split2/train.jsonl" \
"datasets/gwilliams2023/preprocess5/split2/val.jsonl" \
"datasets/gwilliams2023/preprocess5/split2/test.jsonl" \
 --language="English"