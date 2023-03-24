# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import argparse
import sys
import os
import json
from random import shuffle
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

import os
import pickle

from CPC_audio.cpc.feature_loader import buildFeature, FeatureModule, loadModel
from CPC_audio.cpc.dataset import findAllSeqs, filterSeqs
from dataset import AudioDataset
from model import NNESKMeans

import torchaudio

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def getQuantile(sortedData, percent):
    return sortedData[int(percent * len(sortedData))]


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpc-path', type=str,
                        help="Path to the checkpoint of CPC module.")
    parser.add_argument('--trained-path', type=str,
                        help="Path to the output clustering checkpoint.")
    parser.add_argument('--pretrained-path', type=str, default=None)
    parser.add_argument('--path-list', type=str)
    parser.add_argument('--extension', type=str, default='.wav')
    parser.add_argument('--max-size-seq', type=int, default=2048000,#10240,
                        help="The size of the window when loading audio data (default: 10240).")
    parser.add_argument('--level_gru', type=int, default=None,
                        help='Specify the LSTM hidden level to take the representation (default: None).')
    parser.add_argument('--encoder_layer', action='store_true',
                        help='Whether to use the output of the encoder for the clustering.')
    parser.add_argument('--strict', type=bool, default=True,
                        help='If activated, each batch of feature '
                        'will contain exactly max_size_seq frames (defaut: True).')
    parser.add_argument('--nneskmeans-inner-batch-size', type=int, default=32)
    #parser.add_argument('--cpc-inner-batch-size', type=int, default=128)
    parser.add_argument('--word-size', type=int, default=1000)
    parser.add_argument('--input-dim', type=int, default=512)
    parser.add_argument('--output-dim', type=int, default=768)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--max-len', type=int, default=100)
    parser.add_argument('--min-len', type=int, default=1)
    #parser.add_argument('--min-dur', type=int, default=1)
    parser.add_argument('--num-epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max-tokens', type=int, default=48000)
    parser.add_argument('--decoder', action='store_true')
    #parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--segment-temp', type=float, default=None)
    parser.add_argument('--cluster-temp', type=float, default=None)
    #', action='store_true')
    #parser.add_argument('--cpc', action='store_true')
    #parser.add_argument('--min-word-len', type=int, default=None)
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--lang', type=str, default=None)
    parser.add_argument('--result-dir', type=str)

    return parser.parse_args(argv)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    args = parseArgs(sys.argv[1:])

    print(args)

    nneskmeans = NNESKMeans(
            word_size=args.word_size, 
            input_dim=args.input_dim, output_dim=args.output_dim, 
            num_layers=args.num_layers, batch_size=args.nneskmeans_inner_batch_size,
            max_len=args.max_len, min_len=args.min_len, is_decoder=args.decoder
            )
    
    print("Loading trained model")
    state_dict = torch.load(args.trained_path)
    nneskmeans.load_state_dict(state_dict["state_dict"], strict=False)
    nneskmeans = nneskmeans.to("cuda").eval()

    words = [[] for _ in range(args.word_size)]

    with open(args.path_list, "r") as f:
        wav_paths = f.read().split('\n')[:-1]
    keys = [p.split('/')[-1].split('.')[0] for p in wav_paths]

    if args.level_gru is None:
        updateConfig = None
    else:
        updateConfig = argparse.Namespace(nLevelsGRU=args.level_gru)

    model = loadModel([args.cpc_path])[0]#, updateConfig=updateConfig)[0]
    featureMaker = FeatureModule(model, args.encoder_layer)

    featureMaker.eval()
    featureMaker.cuda()
    def cpc_feature_function(x):
        return buildFeature(featureMaker, x,
                                seqNorm=False,
                                strict=args.strict)

    total_len = 0
    total_sample_segment = 0
    lens = [0]*(args.max_len+1)
    unsup_landmarks = {}
    lexical_scores = []
    print("Segmenting and Clustering", len(keys), "sequences")
    for j, key in enumerate(keys):
        #print(key, end=" ", flush=True)
        batch_size = args.nneskmeans_inner_batch_size

        feat = cpc_feature_function(wav_paths[j])[0]

        for i in range(int(len(feat)/batch_size) + 1):
            with torch.no_grad():
                if len(feat) - (i+1)*batch_size < args.min_len:
                    feat_i = feat[i*batch_size:][:, :args.input_dim]
                else:
                    feat_i = feat[i*batch_size:(i+1)*batch_size][:, :args.input_dim]
                results, ass, boundaries, segs = nneskmeans(
                            feat_i, 
                            (i*batch_size, (i+1)*batch_size), 
                            args.max_len, train=False, encoder_grad=False, decoder_grad=False,
                            ini_seg=False, downsample=args.downsample)

                total_len += results["total_len"]
                total_sample_segment += results["sample_segment"]

                name = "_".join(key.split("_")[:-1])
                start = key.split("_")[-1].split("-")[0]
                segs = (torch.stack(segs) + float(start)) * 0.01
                for idx, (s, e) in zip(ass, segs):
                    words[idx].append(name+" "+str(s.item())+" "+str(e.item()))

    print()
    print("average lenght of segments : ", total_len/total_sample_segment)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    result_path = os.path.join(args.result_dir, args.lang+".txt")
    if os.path.exists(result_path):
        os.remove(result_path)
    with open(result_path, "w") as f:
        for idx in range(args.word_size):
            f.write("Class "+str(idx)+"\n")
            for word in words[idx]:
                f.write(word+"\n")
            f.write("\n")
    print("result is saved at : ", result_path)

    