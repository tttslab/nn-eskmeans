import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import argparse
import sys
import os
import json
import random
from random import shuffle
from pathlib import Path

import scipy.signal as signal

import pickle
import tqdm

from sklearn.cluster import kmeans_plusplus

from torch.utils.tensorboard import SummaryWriter

import os

#sys.path.append("CPC_audio")
from CPC_audio.cpc.feature_loader import buildFeature, FeatureModule, loadModel
from CPC_audio.cpc.dataset import findAllSeqs, filterSeqs#AudioBatchData

from dataset import AudioDataset
from model import NNESKMeans

import torchaudio

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpc-path', type=str,
                        help="Path to the checkpoint of CPC module.")
    parser.add_argument('--output-dir', type=str,
                        help="Path to the output model checkpoint.")
    parser.add_argument(
        '--pathDB', type=str,
        default="/datasets01/LibriSpeech/022219/train-clean-100/")
    parser.add_argument(
        '--pretrained-path', type=str, default=None)
    parser.add_argument(
        '--path-list', type=str)
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
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--max-tokens', type=int, default=48000)
    parser.add_argument('--update-max-tokens', type=int, default=48000)
    parser.add_argument('--decoder', action='store_true')
    #parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--segment-temp', type=float, default=None)
    parser.add_argument('--cluster-temp', type=float, default=None)
    #', action='store_true')
    parser.add_argument('--train-AE', action='store_true')
    parser.add_argument('--AE-grad', action='store_true')
    parser.add_argument('--encoder-grad', action='store_true')
    parser.add_argument('--decoder-grad', action='store_true')
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--lr-step', type=int, default=10)
    parser.add_argument('--warm-step', type=int, default=100)
    #parser.add_argument('--cpc', action='store_true')
    #parser.add_argument('--min-word-len', type=int, default=None)
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--lang', type=str, default=None)
    parser.add_argument('--loss-weight', type=float, default=0.25)

    return parser.parse_args(argv)

def kmeans_plusplus(embs, word_size):
    num_embs, dim = embs.size()

    if num_embs < word_size:
        print("sample size is smaller than word_size")
        exit()

    square_embs = torch.sum(embs**2, dim=-1)

    centers = torch.zeros(word_size, dim).to("cuda")
    idx_list = []

    idx = random.randint(0, num_embs-1)
    idx_list.append(idx)

    centers[0] = embs[idx]

    closest_dist = torch.sum((embs - centers[0])**2, dim=-1)
    weights = closest_dist / torch.sum(closest_dist)
    weights[idx] = 0

    for i in range(1, word_size):
        gumbels = (-torch.empty_like(weights, memory_format=torch.legacy_contiguous_format).exponential_().log())
        _, idx = torch.max(torch.log(weights) + gumbels, dim=0)
        idx_list.append(idx)

        centers[i] = embs[idx]

        dist = torch.sum((embs - centers[i])**2, dim=-1)
        dist[idx] = 0

        replace_idxs =  dist < closest_dist
        closest_dist[replace_idxs] = dist[replace_idxs]
        weights = closest_dist / torch.sum(closest_dist)

    return centers, idx_list

def main():

    torch.cuda.empty_cache()
    args = parseArgs(sys.argv[1:])

    print(args)

    slice_len = args.max_len
    torch.manual_seed(3047)

    LANG = args.lang
    dataset = AudioDataset(args.path_list)

    def simple_collate(batch):
        return batch
    trainLoader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=simple_collate)
    print(f"Length of dataLoader: {len(trainLoader)}")

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
    print("CPC loaded!")

    # Check if dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    nneskmeans = NNESKMeans(
            word_size=args.word_size, 
            input_dim=args.input_dim, output_dim=args.output_dim, 
            num_layers=args.num_layers, batch_size=args.nneskmeans_inner_batch_size,
            max_len=args.max_len, min_len=args.min_len, is_decoder=args.decoder
            ).to("cuda")

    start_epoch = 1
    log_i = 0
    if args.pretrained_path is not None:
        print("loading pretrained model")
        
        state_dict = torch.load(args.pretrained_path)

        nneskmeans_load = NNESKMeans(
            word_size=args.word_size, 
            input_dim=args.input_dim, output_dim=args.output_dim, 
            num_layers=args.num_layers, batch_size=args.nneskmeans_inner_batch_size,
            max_len=args.max_len, min_len=args.min_len, is_decoder=args.decoder
            ).to("cuda")

        nneskmeans_load.load_state_dict(state_dict["state_dict"])
        nneskmeans.encoder = nneskmeans_load.encoder
        nneskmeans.decoder = nneskmeans_load.decoder

        del nneskmeans_load
        
        nneskmeans = nneskmeans.to("cuda")

    def random_extract(min_len, max_len):
        nneskmeans.eval()
        embs_list = []
        durs_list = []
        with torch.no_grad():
            if LANG == "mandarin":
                iter_num = 15
            else:
                iter_num = 1
            for _ in range(iter_num):
                for data in trainLoader:
                    feats, _ = data[0] 

                    feats = cpc_feature_function(feats)[0]

                    seq_len = len(feats)

                    durs = torch.randint(low=min_len-1, high=max_len+1, size=(seq_len,))

                    landmark = 0
                    feat_segment = []
                    for dur in durs:
                        if landmark+dur >= seq_len:
                            feat_segment.append(feats[landmark:])
                            break
                        else:
                            feat_segment.append(feats[landmark:landmark+dur])
                        landmark += dur
                    feats = feat_segment
                    del feat_segment
                    durs = durs[:len(feats)]
                    if len(feats[-1]) > min_len-1:
                        durs[-1] = len(feats[-1])
                    else:
                        if len(durs)<2:
                            continue
                        durs = durs[:-1]
                        feats = feats[:-1]

                    if args.downsample:
                        embs = []
                        for f in feats:
                            f = f.numpy()
                            x = signal.resample(f, 10, axis=0).flatten()
                            embs.append(x)
                        embs = np.array(embs)
                        embs = torch.from_numpy(embs)
                    else:
                        x = nneskmeans.encoder(feats, pack=True)[0]
                        embs = []
                        for x_i, durs_i in zip(x, durs):
                            x_i = x_i[durs_i - 1]
                            embs.append(x_i)
                        embs = torch.stack(embs)

                    embs_list.extend(embs)
                    durs_list.extend(durs)
                
            embs = torch.stack(embs_list)
            durs = torch.stack(durs_list)

            return embs, durs
    
    # sampling word emb from output of encoder that encording sylseg feat
    if not args.train_AE:
        print("Extracting sample features")
        min_len = args.min_len
        
        embs, durs = random_extract(min_len, args.max_len)

        print("Sampling "+str(args.word_size)+" inital centers from "+str(len(embs))+" samples by kmeans++")

        with torch.no_grad():
            embs, idx_list = kmeans_plusplus(embs.to("cuda"), args.word_size)

        nneskmeans.word_emb = nn.Parameter(embs)
        nneskmeans.word_size = args.word_size
        del embs
    nneskmeans.temp = args.segment_temp
    nneskmeans = nneskmeans.to("cuda")

    def func(epoch_i):
        min_rate = 0.01
        if epoch_i < args.warm_step:
            #rate =  epoch_i / args.warm_step
            rate =  min_rate + (1 - min_rate) / args.warm_step * (epoch_i - 1)
        else:
            if args.lr_step < 1:
                return 1
            else:
                rate = 0.9**(int(epoch_i/args.lr_step) - 1)

        if rate < min_rate:
            return min_rate
        else:
            return rate

    optimizer = torch.optim.Adam(nneskmeans.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = func)

    if args.pretrained_path is not None:
        if "optimizer_state_dict" in list(state_dict):
            print("loading optimizer state_dict")
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cuda")

    out_state_dict = {}
    out_state_dict["state_dict"] = nneskmeans.state_dict()
    out_state_dict["optimizer_state_dict"] = optimizer.state_dict()
    out_state_dict["epoch"] = start_epoch
    out_state_dict["args"] = args
    out_state_dict["log_i"] = log_i
    torch.save(out_state_dict, os.path.join(args.output_dir, "checkpoint0.pt"))

    num_batch = len(trainLoader)

    writer = SummaryWriter(log_dir=args.log_dir)

    start_time = time.time()
    segment_loss = 0
    rec_loss = 0
    sample_segment = 0
    sample_rec = 0
    log_seg_loss = 0
    log_rec_loss = 0
    mean = 0
    log_codebook_diversity = 0
    codebook = torch.zeros(args.word_size)
    log_mean = 0
    log_num_words = 0
    log_num_segment = 0
    token_count = 0
    count = 0
    update_count = 0
    total_len = 0
    total_sample_segment = 0
    total_seg_loss = 0
    total_rec_loss = 0

    min_loss = float("inf")
    min_model = None

    nneskmeans.train()
    if not args.train_AE:
        if not args.encoder_grad:
            nneskmeans.encoder.eval()
        if not args.decoder_grad:
            nneskmeans.decoder.eval()

    print("train start")
    for epoch in range(start_epoch, args.num_epoch):
        prev_embs = nneskmeans.word_emb.detach().clone()

        num_data = len(trainLoader)
        for index, data in enumerate(trainLoader):
            data, seglist = data[0]

            data = cpc_feature_function(data)[0]

            token_count += len(data)
            if token_count>args.update_max_tokens:
                total_loss = segment_loss + rec_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                update_count += 1

                del total_loss
                mean /= count

                segment_loss = segment_loss.detach().item()
                rec_loss = rec_loss.detach().item()
                
                log_seg_loss += segment_loss*(sample_segment)
                log_rec_loss += rec_loss*sample_rec
                log_mean += mean*count
                log_num_segment += sample_segment

                segment_loss = 0
                rec_loss = 0
                sample_segment = 0
                sample_rec = 0
                token_count = 0
                count = 0
                mean = 0

            count += 1

            if args.train_AE:
                results, _, _, _ = nneskmeans.forward_AE(data, slice_len)
            else:
                results, _, _, _ = nneskmeans(
                        data, seglist, slice_len, train=True, 
                        encoder_grad=args.encoder_grad, decoder_grad=args.decoder_grad,
                        ini_seg=False, downsample=args.downsample,
                        segment_temp=args.segment_temp, cluster_temp=args.cluster_temp, loss_weight=args.loss_weight)

            segment_loss += results["segment_loss"]/args.output_dim
            rec_loss += results["rec_loss"]/args.output_dim
            sample_segment += results["sample_segment"]
            sample_rec += results["sample_rec"]

            mean += results["mean"]
            if results["codebook_diversity"] is None:
                codebook = None
            else:
                codebook += results["codebook_diversity"]
            total_len += results["total_len"]
            total_sample_segment += results["sample_segment"]
            total_seg_loss += results["segment_loss"].detach().item()/args.output_dim
            total_rec_loss += results["rec_loss"].detach().item()/args.output_dim

        if codebook is not None:
            num_words = (codebook > 0).int().sum().item()
            #print(codebook_diversity[:10])
            max_idx = torch.argsort(codebook, dim=0)
            max_nums = [str(int(c.item())) for c in codebook[max_idx[-5:]]]
            max_nums = ' '.join(max_nums)
            codebook_diversity = codebook / codebook.sum()
            codebook_diversity = codebook_diversity[(codebook_diversity > 0)]
            codebook_diversity = (-1)*torch.mean(codebook_diversity*torch.log(codebook_diversity + 1e-12)).item()
        else:
            num_words = -1
            max_nums = str(-1)
            codebook_diversity = -1

        writer.add_scalar("segment_loss", total_seg_loss, log_i)
        writer.add_scalar("rec_loss", total_rec_loss, log_i)
        writer.add_scalar("mean", total_len/total_sample_segment, log_i)
        writer.add_scalar("codebook_diversity", codebook_diversity, log_i)
        writer.add_scalar("num_words", num_words, log_i)
        log_i += 1

        codebook = torch.zeros(args.word_size)

        dif = torch.sum((prev_embs - nneskmeans.word_emb.detach())**2).item()

        scheduler.step()

        print(
            str(epoch)+
            " | segment_loss:"+str(total_seg_loss)+
            " | rec_loss:"+str(total_rec_loss)+
            " | word_dif:"+str(dif)+
            " | mean:"+str(total_len/total_sample_segment)+
            " | codebook_diversity:"+str(codebook_diversity)+
            " | num_words:"+str(num_words)+
            "("+max_nums+")"+
            " | lr:"+str(scheduler.get_last_lr()[0])+
            " | num_segment:"+str(total_sample_segment)+
            " | num_update:"+str(update_count)
            )

        out_state_dict = {}
        out_state_dict["state_dict"] = nneskmeans.state_dict()
        out_state_dict["optimizer_state_dict"] = optimizer.state_dict()
        out_state_dict["epoch"] = int(epoch)
        out_state_dict["args"] = args
        out_state_dict["log_i"] = log_i
        if os.path.exists(os.path.join(args.output_dir, "checkpoint"+str(epoch-10)+".pt")) and not (((epoch-10)%args.save_epoch) == 0):
            os.remove(os.path.join(args.output_dir, "checkpoint"+str(epoch-10)+".pt"))
        torch.save(out_state_dict, os.path.join(args.output_dir, "checkpoint"+str(epoch)+".pt"))

        if args.train_AE:
            loss = total_rec_loss
        else:
            loss = total_seg_loss
        if min_loss > loss:
            model_name = "checkpoint"+str(epoch)+"_"+str(round(loss, 2))+".pt"
            torch.save(out_state_dict, os.path.join(args.output_dir, model_name))
            if (min_model is not None) and (os.path.exists(os.path.join(args.output_dir, min_model))):
                os.remove(os.path.join(args.output_dir, min_model))
            min_loss = loss
            min_model = model_name

        log_seg_loss = 0
        log_rec_loss = 0
        log_mean = 0
        log_codebook_diversity = 0
        log_num_words = 0
        log_num_segment = 0
        total_len = 0
        total_sample_segment = 0
        total_seg_loss = 0
        total_rec_loss = 0

    writer.close()

if __name__ == "__main__":
    main()