import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_sequence, pack_padded_sequence

import copy

import numpy as np
import scipy.signal as signal

class NNESKMeans(nn.Module):
    def __init__(self, 
            word_size, input_dim, output_dim, 
            num_layers, batch_size,
            max_len, min_len,
            is_decoder=False
            ):
        super(NNESKMeans, self).__init__()

        self.batch_size = batch_size
        self.max_len = max_len
        self.min_len = min_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.word_size = word_size
        self.is_decoder = is_decoder

        self.word_emb = nn.Parameter(
                1*torch.Tensor(word_size, output_dim).uniform_(-1, 1)
        )

        self.encoder = LSTMEncoder(input_dim, output_dim, num_layers)

        self.decoder = LSTMDecoder(output_dim, input_dim, num_layers, word_size)
    
    def get_embs(self, feats, downsample=False):
        if downsample:
            embs = []
            for f in feats:
                embs_i = []
                f = f.numpy()
                for i in range(1, len(f)+1):
                    x = signal.resample(f[:i], 10, axis=0).flatten()
                    embs_i.append(x)
                embs_i = torch.from_numpy(np.array(embs_i))
                embs.append(embs_i)
            embs = pad_sequence(embs, batch_first=True, padding_value=0.0).to("cuda")
            #embs = torch.from_numpy(embs).to("cuda")
        else:
            embs = self.encoder(feats, pack=True)[0]

        return embs

    def forward(self, 
        feats, seglist, slice_len, 
        train=True, encoder_grad=True, decoder_grad=True, ini_seg=True, 
        downsample=False,
        segment_temp=None, cluster_temp=None, loss_weight=0.25,
        emb_start_idx=0, emb_end_idx=None
        ):
        seq_len = len(feats)
        if emb_end_idx is None:
            emb_end_idx = self.word_size

        with torch.no_grad():
            if ini_seg:
                if train:
                    with torch.no_grad():
                        x = self.encoder(feats, pack=True)[0]
                else:
                    x = []
                    for i in range(int(len(feats)/self.batch_size)+1):
                        feats_i = feats[i*self.batch_size  : (i+1)*self.batch_size]
                        x.extend(self.encoder(feats_i, pack=True)[0])
                
                durs = []
                for seg in seglist:
                    durs.append(seg[:,1] - seg[:,0])

                embs = []
                for x_i, durs_i in zip(x, durs):
                    x_i = x_i[durs_i - 1]
                    embs.append(x_i)

                durs = torch.stack(durs) 
            else:
                start_land = seglist[0]
                feats = torch.cat([feats, feats.new_zeros(self.max_len-1, self.input_dim)])
                embs = []
                seglist = []
                i=-1

                for i in range(int(seq_len/self.batch_size)):
                    feats_i = [
                            feats[i*self.batch_size+j:i*self.batch_size+j+self.max_len] 
                            for j in range(self.batch_size)
                            ]
                    with torch.no_grad():
                        #embs.extend(self.encoder(feats_i, pack=True)[0])
                        embs.extend(self.get_embs(feats_i, downsample))
                    seglist.extend([list(range(start_land+i*self.batch_size+j, start_land+i*self.batch_size+j+self.max_len)) for j in range(self.batch_size)])
                feats_i = [
                        feats[(i+1)*self.batch_size+j:(i+1)*self.batch_size+j+self.max_len] 
                        for j in range(seq_len-(i+1)*self.batch_size)
                        ]
                if not (len(feats_i)==0):
                    with torch.no_grad():
                        #x = self.encoder(feats_i, pack=True)[0]
                        #embs.extend(self.encoder(feats_i, pack=True)[0])
                        embs.extend(self.get_embs(feats_i, downsample))
                seglist.extend([list(range(start_land+(i+1)*self.batch_size+j, start_land+(i+1)*self.batch_size+j+self.max_len)) for j in range(seq_len-(i+1)*self.batch_size)])
                seglist = torch.unsqueeze(torch.tensor(seglist), -1)
                seglist = torch.cat([
                    seglist[:, 0, :].expand(self.max_len, len(seglist), 1).permute(1,0,2), 
                    seglist+1]
                    , dim=2)

                durs = seglist[:, :, 1] - seglist[:, :, 0] 

            embs = torch.stack(embs).view(-1, self.output_dim)
            durs = durs.flatten().to("cuda")

            segment_loss = 0
            rec_loss = torch.tensor(0).to(torch.float).to("cuda")

            sample_segment = 0
            sample_rec = 0
            sample_token = 0

            len_mean = 0

            codebook_diversity = torch.zeros(len(self.word_emb[emb_start_idx:emb_end_idx]))

            #seq_len = int(len(embs)/slice_len)
            # get distance matrix
            distances = []

            if train:
                with torch.no_grad():
                    d_i = []
                    for i in range(int(len(embs)/self.batch_size)+1):
                        d_prob_i = (torch.sum(embs[i*self.batch_size  : (i+1)*self.batch_size]**2, dim=-1, keepdim=True) 
                                + (-2)*torch.matmul(embs[i*self.batch_size  : (i+1)*self.batch_size], self.word_emb[emb_start_idx:emb_end_idx].T) 
                                + torch.sum(self.word_emb[emb_start_idx:emb_end_idx]**2, dim=-1))

                        d_prob_i = torch.min(d_prob_i, dim=-1)[1]       

                        d_i.extend(d_prob_i)
                    d_i = torch.stack(d_i)

                    d_i_gumbel = d_i
                d = torch.sum((embs - self.word_emb[emb_start_idx:emb_end_idx][d_i])**2, dim=-1)
            else:
                d = []
                d_i = []
                for i in range(int(len(embs)/self.batch_size)+1):
                    dev_seq = embs[i*self.batch_size  : (i+1)*self.batch_size]
                    d_dev = torch.sum(dev_seq**2, dim=-1, keepdim=True)  - 2*torch.matmul(dev_seq, self.word_emb.T) + torch.sum(self.word_emb**2, dim=-1)
                    d_dev, d_i_dev = torch.min(d_dev, dim=-1)

                    d.append(d_dev)
                    d_i.append(d_i_dev)
                
                d = torch.cat(d, dim=-1)
                d_i = torch.cat(d_i, dim=-1)

            distances = d*durs

            del d
            #del durs
            distances = distances.view(-1, slice_len)
            ass = d_i.view(-1, slice_len)
            seglist = seglist.view(-1, slice_len, 2)

            embs = embs.view(-1, self.max_len, self.output_dim)
            durs = durs.view(-1, self.max_len)

            if not decoder_grad:
                del embs
                del durs

            if not len(distances)==seq_len:
                print("batch division error")
                exit()
            d_matrix = float("inf")*distances.new_ones(seq_len+1, seq_len+self.max_len - 1)
            for i in range(seq_len):d_matrix[i][i+self.min_len-1:i+self.max_len] = distances[i][self.min_len-1:self.max_len] # matrix[a,b] is score(seq[a:b+1])
            id_matrix = -1*torch.ones(seq_len+1, seq_len+self.max_len - 1, dtype=torch.long)
            for i in range(seq_len):id_matrix[i][i+self.min_len-1:i+self.max_len] = ass[i][self.min_len-1:self.max_len] # matrix[a,b] is score(seq[a:b+1])
            seglist_matrix = -1*torch.ones(seq_len+1, seq_len+self.max_len - 1, 2, dtype=torch.long)
            for i in range(seq_len):seglist_matrix[i][i+self.min_len-1:i+self.max_len] = seglist[i][self.min_len-1:self.max_len] # matrix[a,b] is score(seq[a:b+1])

            # if seq len is smaller than min_len
            if seq_len < self.min_len:
                whole_seg = seglist[0][seq_len - 1]
                whole_ass = ass[0][seq_len - 1]

            gammas = float('inf')*distances.new_ones(seq_len+1)
            gammas[0] = 0
            boundaries = [0]*(seq_len+1)
            boundaries[0] = [0]
            ass = [0]*(seq_len+1)
            ass[0] = [0]
            seglist = [0]*(seq_len+1)
            seglist[0] = [(-1, -1)]

            mean_pen = []
            for t in range(self.min_len, seq_len+1):

                distances = d_matrix.T[t-1]
                
                distances = distances + gammas
                
                if train and (segment_temp is not None):
                    #min_d, b = torch.min(distances, dim=-1)
                    gumbels = (
                            -torch.empty_like(distances, memory_format=torch.legacy_contiguous_format).exponential_().log()
                        )
                    _, b_g = torch.max(-1*(distances)/segment_temp + gumbels, dim=-1)
                    min_d_g = distances[b_g]
                    min_d = min_d_g #- (min_d_g - min_d).detach()
                    b = b_g
                else:
                    min_d, b = torch.min(distances, dim=-1)

                gammas[t] = min_d 
                ass[t] = ass[b].copy()
                ass[t].append(id_matrix[b][t-1].item())
                seglist[t] = seglist[b].copy()
                seglist[t].append(seglist_matrix[b][t-1])
                boundaries[t] = boundaries[b].copy()
                boundaries[t].append(t) #0含めてtつ目


            del d_matrix
            del seglist_matrix
            del id_matrix
            boundaries = boundaries[-1]
            seglist = seglist[-1]
            ass = ass[-1]

            if seq_len < self.min_len:
                ass = [0, whole_ass]
                seglist = [(-1, -1), whole_seg]
                boundaries = [0, whole_seg[0], whole_seg[1]]

        # recompute for grad
        if train:
            feat_for_grad = []
            durs = []
            for seg in seglist[1:]:
                feat_for_grad.append(feats[seg[0] : seg[1]])
                durs.append(seg[1] - seg[0])
            durs = torch.stack(durs).to("cuda")

            if encoder_grad:
                x = self.get_embs(feat_for_grad, downsample)
            else:
                with torch.no_grad():
                    x = self.get_embs(feat_for_grad, downsample)

            embs = []
            for x_i, durs_i in zip(x, durs):
                x_i = x_i[durs_i - 1]
                embs.append(x_i)
            embs = torch.stack(embs)
                            
            if cluster_temp is not None:
                with torch.no_grad():
                    d_prob_i = (torch.sum(embs**2, dim=-1, keepdim=True) 
                                        + (-2)*torch.matmul(embs, self.word_emb[emb_start_idx:emb_end_idx].T) 
                                        + torch.sum(self.word_emb[emb_start_idx:emb_end_idx]**2, dim=-1))
                gumbels = (
                    -torch.empty_like(d_prob_i, memory_format=torch.legacy_contiguous_format).exponential_().log()
                )
                d_prob_i = torch.max(-d_prob_i/cluster_temp + gumbels, dim=-1)[1]
            else:
                d_prob_i = ass[1:]

            d = (torch.sum((embs.detach() - self.word_emb[emb_start_idx:emb_end_idx][d_prob_i])**2, dim=-1) + loss_weight*torch.sum((embs - self.word_emb[emb_start_idx:emb_end_idx][d_prob_i].detach())**2, dim=-1))/(1+loss_weight)

            d = d * durs
            segment_loss = torch.sum(d)/(self.max_len/2)
        else:
            segment_loss = gammas[-1]/(self.max_len/2)
        
        del gammas
        sample_segment = len(seglist[1:])

        sample_token = 0

        output_idx = [(e-s).item() for s, e in seglist[1:]]
        output_idx = torch.tensor(output_idx)
        output_idx = output_idx.to(torch.float)
        mean = torch.mean(output_idx).item()

        len_mean += mean
        for i in ass[1:]:
            codebook_diversity[i] += 1
        total_len = sum(output_idx).item()

        # train decoder
        if train and decoder_grad:
            #embs = self.word_emb[d_prob_i] + (embs - self.word_emb[d_prob_i]).detach()

            if ini_seg:
                durs = [e-s for s, e in seglist[1:]]
                feats = [feats[s][:d] for s, d in zip(boundaries[:-1], durs)]
            else:
                segs = torch.stack(seglist[1:]) - start_land
                durs = segs[:, 1] - segs[:, 0]
                feats = [feats[start:end] for start, end in segs]

            for i in range(int(len(embs)/self.batch_size)+1):
                e = embs[i*self.batch_size:(i+1)*self.batch_size]
                lens = durs[i*self.batch_size:(i+1)*self.batch_size]
                if len(lens)==0:break

                x_rec = self.decoder(e, lens)

                x = feats[i*self.batch_size:(i+1)*self.batch_size]

                x = pad_sequence(x, batch_first=True, padding_value=0.0).to("cuda")

                x = torch.sum((x - x_rec)**2, dim=-1)

                rec_loss += sum([sum(x[j][:lens[j]]) for j in range(len(lens))])
     
        results = {
                "segment_loss":segment_loss,
                "rec_loss":rec_loss,
                "sample_segment":sample_segment,
                "sample_rec":sample_rec,
                "sample_token":sample_token,
                "mean":len_mean,
                "total_len":total_len,
                "codebook_diversity":codebook_diversity
                }

        return results, ass[1:], boundaries, seglist[1:]

    def forward_AE(self, feats, slice_len):
        seq_len = len(feats)

        durs = torch.randint(low=self.min_len, high=slice_len+1, size=(seq_len,))

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

        if len(feats[-1]) > self.min_len-1:
            durs[-1] = len(feats[-1])
        else:
            durs = durs[:-1]
            feats = feats[:-1]

        x = self.encoder(feats, pack=True)[0]

        embs = []
        for x_i, durs_i in zip(x, durs):
            x_i = x_i[durs_i - 1]
            embs.append(x_i)
        embs = torch.stack(embs)

        sample_segment = len(embs)
        rec_loss = 0
        len_mean = torch.mean(durs.to(torch.float)).item()
        len_sum = torch.sum(durs).item()
        for i in range(int(len(embs)/self.batch_size)+1):
            e = embs[i*self.batch_size:(i+1)*self.batch_size]
            lens = durs[i*self.batch_size:(i+1)*self.batch_size]
            if len(lens)==0:break

            x_rec = self.decoder(e, lens)

            x = feats[i*self.batch_size:(i+1)*self.batch_size]

            x = pad_sequence(x, batch_first=True, padding_value=0.0).to("cuda")

            x = torch.sum((x - x_rec)**2, dim=-1)

            rec_loss += sum([sum(x[j][:lens[j]]) for j in range(len(lens))])
     
        results = {
                "segment_loss":torch.tensor(0),
                "rec_loss":rec_loss,
                "sample_segment":sample_segment,
                "sample_rec":0,
                "sample_token":0,
                "mean":len_mean,
                "total_len":len_sum,
                "codebook_diversity":None
                }

        return results, None, None, None

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = 0.1
        self.dropout_input = nn.Dropout(self.dropout)
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(
                input_dim, hidden_dim, 
                num_layers=num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=False)
        
    def forward(self, x, pack=False):
        x_len = [len(x[j]) for j in range(len(x))]
        x = pad_sequence(x, batch_first=True, padding_value=0.0).to("cuda")
        x = self.dropout_input(x)
        
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0, total_length=None)

        return x, None

class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, word_size):
        super(LSTMDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = 0.1
        self.dropout_input = nn.Dropout(self.dropout)
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(
                hidden_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=False)

        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lens):
        x = self.dropout_input(x)
        x = x.expand(max(lens), x.size(0), x.size(1)).permute(1,0,2)
        
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0, total_length=None)
        
        x = self.proj(x)
        return x
