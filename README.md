# Data prepareation
```
mkdir ../data
```
Download `zerospeech2020.zip` and `2017_vads.zip` under ../data/ from https://zerospeech.com/challenge_archive/2020/data/ .

# Prepocessing
Split audio data based on based on voice activity detection.
```
python scripts/segment_by_vad.py \
    --data-dir ../data/2020/2017/mandarin/train \
    --vad-path ../data/2020/2017/vads/MANDARIN_VAD.csv \
    --lang mandarin
```

# Training

## CPC
Train CPC model using the implimentation at https://github.com/tuanh208/CPC_audio/tree/zerospeech .

```
git clone https://github.com/tuanh208/CPC_audio/tree/zerospeech

mkdir -p CPC_audio/checkpoints/mandarin

python cpc/train.py \
    --pathDB ../../data/2020/2017/mandarin/train/segment_by_vad \
    --pathCheckpoint CPC_audio/checkpoints/mandarin \
    --pathTrain ../../data/2020/2017/mandarin/train/segment_by_vad/keylist_mandarin_segment_by_vad.txt \
    --pathVal ../../data/2020/2017/mandarin/train/segment_by_vad/keylist_mandarin_segment_by_vad_val.txt \
    --file_extension .wav --ignore_cache \
    --hiddenEncoder 512 --hiddenGar 512 --nLevelsGRU 4 --batchSizeGPU 16 --dropout --schedulerRamp 100 \
    --nEpoch 5000 --save_step 100 --n_process_loader 8
```

## Encoder pretraining
```
mkdir -p checkpoints/mandarin

python scripts/train.py \
    --cpc-path CPC_audio/checkpoints/mandarin/checkpoint_0.pt \
    --output-dir checkpoints/mandarin/AE \
    --path-list ../data/2020/2017/mandarin/train/segment_by_vad/mandarin_segment_by_vad.txt
    --extension .wav \
    --level_gru 2 \
    --max-tokens 12000 --update-max-tokens 144000 \
    --nneskmeans-inner-batch-size  2048 \
    --word-size 500 --min-len 1 --max-len 50 \
    --num-epoch 5000 --lr 0.005 --save-epoch 250 --lr-step 250 --warm-step 250\
    --log-interval 10 --log-dir checkpoints/mandarin/AE/log --input-dim 512 --output-dim 512 \
    --lang mandarin --train-AE
```

## NN-ES-KMeans
```
python scripts/train.py \
    --cpc-path CPC_audio/checkpoints/mandarin/checkpoint_0.pt \
    --output-dir checkpoints/mandarin/nneskmeans \
    --pretrained-path checkpoints/mandarin/AE/checkpoint0.pt \
    --path-list ../data/2020/2017/mandarin/train/segment_by_vad/mandarin_segment_by_vad.txt \
    --extension .wav \
    --level_gru 2 \
    --max-tokens 12000 --update-max-tokens 144000 \
    --nneskmeans-inner-batch-size  2048 \
    --word-size 500 --min-len 15 --max-len 50 \
    --num-epoch 1000 --lr 0.0001 --save-epoch 10 --lr-step 0 --warm-step 10\
    --log-interval 10 --log-dir checkpoints/mandarin/nneskmeans/log --input-dim 512 --output-dim 512 \
    --lang mandarin --loss-weight 0.25 --encoder-grad
    # --segment-temp 2 --cluster-temp 0.1
```

# Evaluation
```
python scripts/eval.py \
    --cpc-path CPC_audio/checkpoints/mandarin/checkpoint_0.pt \
    --trained-path checkpoints/mandarin/nneskmeans/checkpoint1.pt \
    --result-dir results/mandarin/2017/track2 \
    --path-list ../data/2020/2017/mandarin/train/segment_by_vad/mandarin_segment_by_vad.txt \
    --extension .wav \
    --level_gru 2 \
    --nneskmeans-inner-batch-size  2048 \
    --word-size 500 --min-len 15 --max-len 50 \
    --input-dim 512 --output-dim 512 --lang mandarin
```

Evaluate using zerospeech2020 evaluation toolkit: https://github.com/zerospeech/zerospeech2020
```
zerospeech2020-evaluate 2017-track2 results/mandarin -l mandarin -o results/mandarin/result.json
~~~
