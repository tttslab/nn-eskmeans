#

# prepocess
mkdir -p checkpoints/mandarin

# cpc
cd CPC_audio
mkdir -p CPC_audio/checkpoints/mandarin

https://github.com/tuanh208/CPC_audio/tree/zerospeech

/net/papilio/storage1/yiwamoto/data/2020/2017/mandarin/train/segment_by_vad

lang=mandarin
PATH_CHECKPOINT_DIR=/net/papilio/storage1/yiwamoto/CPC_audio/cpc/checkpoints/${lang}
TRAINING_SET=/net/papilio/storage1/yiwamoto/clustering/code/keylist_${lang}_segment_by_vad.txt
VAL_SET=/net/papilio/storage1/yiwamoto/clustering/code/keylist_${lang}_segment_by_vad_val.txt
EXTENSION=.wav

python cpc/train.py \
    --pathDB /net/papilio/storage1/yiwamoto/data/${lang}/segment_by_vad/devpart1 \
    --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --ignore_cache \
    --hiddenEncoder 512 --hiddenGar 512 --nLevelsGRU 4 --batchSizeGPU 16 --dropout --schedulerRamp 100 \
    --nEpoch 5000 --save_step 100 --n_process_loader 8

# AE pretrain