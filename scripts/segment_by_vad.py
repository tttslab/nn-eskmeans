import pickle
import os
import librosa
import numpy as np
import sys
import argparse

import random

import pkg_resources


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--vad-path', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--extension', type=str, default='.wav')

    return parser.parse_args(argv)

def main():

    args = parseArgs(sys.argv[1:])

    keylist = []
    for filename in os.listdir(args.data_dir):
        if args.extension in filename:
            keylist.append(filename.rstrip(args.extension))

    with open(args.vad_path, "r") as f:
        vads = f.readlines()

    segments = {}
    for key in keylist:
        segments[key] = []

    for vad in vads[1:-1]:
        vad = vad.strip().split(",")
        key, start, end = vad
        if key in keylist:
            if float(start) < float(end):
                segments[key].append((start, end))

    separate_dir = os.path.join(args.data_dir, "segment_by_vad")
    if not os.path.exists(separate_dir):
        os.makedirs(separate_dir)

    for key in keylist:
        if not os.path.exists(os.path.join(separate_dir, key)):
            os.makedirs(os.path.join(separate_dir, key))

    wav_paths = []
    segmented_keylist = []
    mfcc_cmvn_dd_segmented = {}
    for key in keylist:
        wav_path =  os.path.join(args.data_dir, key+".wav")
        x, sr = librosa.load(wav_path, sr=16000)

        for i, (start, end) in enumerate(segments[key]):
            label = key+"_"+str(round(float(start)*100))+"-"+str(round(float(end)*100))

            separate_path = os.path.join(separate_dir, key, label+".wav")
            segmented_keylist.append(label)
            wav_paths.append(separate_path)

            librosa.output.write_wav(separate_path, x[round(float(start)*sr):round(float(end)*sr)], sr=16000)
    print("save wav files under "+separate_dir)

    wav_paths_path = os.path.join(args.data_dir, "segment_by_vad", args.lang+"_segment_by_vad.txt")
    with open(wav_paths_path, "w") as f:
        for wav_path in wav_paths:
            f.write(wav_path+"\n")

    segmented_keylist_path = os.path.join(args.data_dir, "segment_by_vad", "keylist_"+args.lang+"_segment_by_vad.txt")
    with open(segmented_keylist_path, 'w') as f:
        for segmented_key in segmented_keylist:
            f.write(segmented_key+"\n")

    segment_len = len(segmented_keylist)
    random.shuffle(segmented_keylist)
    segmented_keylist = segmented_keylist[:round(segment_len*0.01)]
    segmented_keylist_path = os.path.join(args.data_dir, "segment_by_vad", "keylist_"+args.lang+"_segment_by_vad_val.txt")
    with open(segmented_keylist_path, 'w') as f:
        for segmented_key in segmented_keylist:
            f.write(segmented_key+"\n")

if __name__=="__main__":
    main()