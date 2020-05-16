# Evaluate the WERs of open ASR APIs

Dependencies:
- iflytek API，see openasr/xunfei/README.md
- google cloud API, see openasr/google/README.md
- baidu API，see openasr/baidu/README.md
- tencent API，see openasr/tencent/README.md


Usage:
- apply the open API and set the secret key in 'openasr/config.py'
- prepare the dataset as testdata/dataset, including 'text', 'wav.scp', 'utt2dur' three files.
- run 'openasr/run_dataset.py' to recognize the wavs and compute WER.
