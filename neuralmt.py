# -*- coding: utf-8 -*-
# Python version: 3
#
# SFU CMPT413/825 Fall 2019, HW4
# default solution
# Simon Fraser University
# Jetic Gū
#
#
import os
import re
import sys
import optparse
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd
from torchtext import data

#import support.hyperparams as hp
#import support.datasets as ds

# hyperparameters
class hp:
    # vocab
    unk_idx = 0
    pad_idx = 1
    sos_idx = 2
    eos_idx = 3

    # architecture
    hidden_dim = 256
    embed_dim = 256
    n_layers = 2
    dropout = 0.2
    batch_size = 32
    num_epochs = 10
    lexicon_cap = 25000

    # training
    max_lr = 1e-4
    cycle_length = 3000

    # generation
    max_len = 50

    # system
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---YOUR ASSIGNMENT---
# -- Step 1: Baseline ---
# The attention module is completely broken now. Fix it using the definition
# given in the HW description.
class AttentionModule(nn.Module):
    def __init__(self, attention_dim):
        """
        You shouldn't deleted/change any of the following defs, they are
        essential for successfully loading the saved model.
        """
        super(AttentionModule, self).__init__()
        self.W_enc = nn.Linear(attention_dim, attention_dim, bias=False)
        self.W_dec = nn.Linear(attention_dim, attention_dim, bias=False)
        self.V_att = nn.Linear(attention_dim, 1, bias=False)
        return

    # Start working from here, both 'calcAlpha' and 'forward' need to be fixed
    def calcAlpha(self, decoder_hidden, encoder_out):
        """
        param encoder_out: (seq, batch, dim),
        param decoder_hidden: (seq, batch, dim)
        """
        seq, batch, dim = encoder_out.shape
        query = self.W_dec(decoder_hidden) # (1, 1, 256)
        keys = self.W_enc(encoder_out.permute(1, 0, 2)) # (1, seq, 256)
        scores = keys + query
        alpha = torch.softmax(self.V_att(torch.tanh(scores)), dim=1)
        return alpha # (1, seq, 1)

    def forward(self, decoder_hidden, encoder_out):
        """
        encoder_out: (seq, batch, dim),
        decoder_hidden: (seq, batch, dim)
        """
        alpha = self.calcAlpha(decoder_hidden, encoder_out)
        seq, _, dim = encoder_out.shape
        # (1, 1, seq) * (1, seq, 256) = (1, 1, 256)
        context = torch.bmm(alpha.permute(0, 2, 1), encoder_out.permute(1, 0, 2))
        return context, alpha.permute(2, 0, 1)


def beamDecoder(decoder, encoder_out, encoder_hidden, maxLen,
                  eos_index, k=20):
    seq1_len, batch_size, _ = encoder_out.size()
    target_vocab_size = decoder.target_vocab_size

    outputs = torch.autograd.Variable(
        encoder_out.data.new(maxLen, batch_size, target_vocab_size))
    # take what we need from encoder
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    # start token (ugly hack)
    output = torch.autograd.Variable(
        outputs.data.new(1, batch_size).fill_(hp.sos_idx).long())

    #first
    output, decoder_hidden, alpha = decoder(
                output, encoder_out, decoder_hidden)
    
    log_out = F.softmax(output, dim=-1).log()

    probs, idxs = log_out[0, 0, :].topk(k, sorted=True)
    idxs = idxs.unsqueeze(-1)
    d_hs = decoder_hidden.repeat(k, 1, 1, 1)
    alphas = torch.zeros(maxLen, k, batch_size, seq1_len)

    for t in range(1, maxLen):
        temptable = torch.zeros(k * k)
        tempidxs = torch.zeros(k * k, t + 1, dtype=torch.int32).to(hp.device)
        tempdhs= torch.zeros(k * k, 2, 1, 256).to(hp.device)
        tempalphas = torch.zeros(k * k, batch_size, seq1_len)
        for bi in range(k):
            output = idxs[bi, t - 1].reshape(1, 1)
            output, decoder_hidden, alpha = decoder(
                output, encoder_out, d_hs[bi])

            log_out = F.softmax(output, dim=-1).log()

            prob, idx = log_out[0, 0, :].topk(k, sorted=True)
            bio = k * (bi)
            temptable[bio: bio+k] = (prob + probs[bi])
            tempidxs[bio: bio+k] = torch.cat([idxs[bi].repeat(k, 1), idx.unsqueeze(-1)], dim=-1) 
            tempdhs[bio:bio+k] = decoder_hidden.repeat(k, 1, 1, 1)
            tempalphas[bio:bio+k] = alpha.repeat(k, 1, 1)

        prob, idx = temptable.topk(k, sorted=True)
        probs = prob
        idxs = tempidxs[idx]
        d_hs = tempdhs[idx]
        alphas[t] = tempalphas[idx]
        idx = torch.argmax(probs)
        if idxs[idx, t] == eos_index:
            break

    idx = torch.argmax(probs)
    tgts = idxs[idx]
    alphas = alphas[:, idx]

    return tgts, alphas

# -- Step 2: Improvements ---
# Implement UNK replacement, BeamSearch, translation termination criteria here,
# you can change 'greedyDecoder' and 'translate'.
def greedyDecoder(decoder, encoder_out, encoder_hidden, maxLen,
                  eos_index):
    seq1_len, batch_size, _ = encoder_out.size()
    target_vocab_size = decoder.target_vocab_size

    outputs = torch.autograd.Variable(
        encoder_out.data.new(0, batch_size, target_vocab_size))
    alphas = torch.zeros(maxLen, batch_size, seq1_len)
    # take what we need from encoder
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    # start token (ugly hack)
    output = torch.autograd.Variable(
        outputs.data.new(1, batch_size).fill_(eos_index).long())
    for t in range(maxLen):
        output, decoder_hidden, alpha = decoder(
            output, encoder_out, decoder_hidden)
        outputs = torch.vstack((outputs, output))
        alphas[t] = alpha.data
        output = torch.autograd.Variable(output.data.max(dim=2)[1])
        if int(output.data) == eos_index:
            break
    return outputs, alphas.permute(1, 2, 0)

def translate(model, test_iter, unk_dict):
    results = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_iter)):
            output, attention = model(batch.src)
            res = []
            unk_i = 0
            for idx, t in enumerate(output):
                if t.data == hp.eos_idx:
                    break
                vals = list(unk_dict[i].values())
                if t.data == hp.unk_idx:
                    attidx = torch.argmax(attention[idx])
                    if attidx.item() in unk_dict[i]:
                        word = unk_dict[i][attidx.item()]
                    else:
                        token = batch.src[attidx, 0].data
                        if token == 0 and len(vals) > unk_i:
                            word = vals[unk_i]
                            if len(vals) > unk_i + 1:
                                unk_i += 1
                        else:
                            word = model.fields['src'].vocab.itos[token]
                    res.append(word)
                else:
                    word = model.fields['tgt'].vocab.itos[t]
                    res.append(word)

            output = []
            prev = None
            for token in res:
                if token != prev:
                    output.append(token)
                    prev = token

            output = ' '.join(output)
            results.append(output)
    return results


# ---Model Definition etc.---
# DO NOT MODIFY ANYTHING BELOW HERE


class Encoder(nn.Module):
    """
    Encoder class
    """
    def __init__(self, source_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(source_vocab_size, embed_dim,
                                  padding_idx=hp.pad_idx)
        self.rnn = nn.GRU(embed_dim,
                          hidden_dim,
                          n_layers,
                          dropout=dropout,
                          bidirectional=True)

    def forward(self, source, hidden=None):
        """
        param source: batched input indices
        param hidden: initial hidden value of self.rnn
        output (encoder_out, encoder_hidden):
            encoder_hidden: the encoder RNN states of length len(source)
            encoder_out: the final encoder states, both direction summed up
                together h^{forward} + h^{backward}
        """
        embedded = self.embed(source)  # (batch_size, seq_len, embed_dim)
        # get encoded states (encoder_hidden)
        encoder_out, encoder_hidden = self.rnn(embedded, hidden)

        # sum bidirectional outputs
        encoder_final = (encoder_out[:, :, :self.hidden_dim] +  # forward
                         encoder_out[:, :, self.hidden_dim:])   # backward

        # encoder_final:  (seq_len, batch_size, hidden_dim)
        # encoder_hidden: (n_layers * num_directions, batch_size, hidden_dim)
        return encoder_final, encoder_hidden


class Decoder(nn.Module):
    def __init__(self, target_vocab_size,
                 embed_dim, hidden_dim,
                 n_layers,
                 dropout):
        super(Decoder, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(target_vocab_size,
                                  embed_dim,
                                  padding_idx=hp.pad_idx)
        self.attention = AttentionModule(hidden_dim)

        self.rnn = nn.GRU(embed_dim + hidden_dim,
                          hidden_dim,
                          n_layers,
                          dropout=dropout)

        self.out = nn.Linear(hidden_dim * 2, target_vocab_size)

    def forward(self, output, encoder_out, decoder_hidden):
        """
        decodes one output frame
        """
        embedded = self.embed(output)  # (1, batch, embed_dim)
        context, alpha = self.attention(decoder_hidden[-1:], encoder_out)
        # 1, 1, 50 (seq, batch, hidden_dim)
        rnn_output, decoder_hidden =\
            self.rnn(torch.cat([embedded, context], dim=2), decoder_hidden)
        output = self.out(torch.cat([rnn_output, context], 2))
        return output, decoder_hidden, alpha


class Seq2Seq(nn.Module):
    def __init__(self, fields=None, srcLex=None, tgtLex=None, build=True):
        super(Seq2Seq, self).__init__()
        # If we are loading the model, we don't build it here
        if build is True:
            self.params = {
                'srcLexSize': len(srcLex.vocab),
                'tgtLexSize': len(tgtLex.vocab),
                'embed_dim': hp.embed_dim,
                'hidden_dim': hp.hidden_dim,
                'n_layers': hp.n_layers,
                'dropout': hp.dropout,
                'fields': fields,
                'maxLen': hp.max_len,
            }
            self.build()

    def build(self):
        # self.params are loaded, start building the model accordingly
        self.encoder = Encoder(
            source_vocab_size=self.params['srcLexSize'],
            embed_dim=self.params['embed_dim'],
            hidden_dim=self.params['hidden_dim'],
            n_layers=self.params['n_layers'],
            dropout=self.params['dropout'])
        self.decoder = Decoder(
            target_vocab_size=self.params['tgtLexSize'],
            embed_dim=self.params['embed_dim'],
            hidden_dim=self.params['hidden_dim'],
            n_layers=self.params['n_layers'],
            dropout=self.params['dropout'])
        self.fields = self.params['fields']
        self.maxLen = self.params['maxLen']

    def forward(self, source, maxLen=None, eos_index=3):
        """
        This method implements greedy decoding
        param source: batched input indices
        param maxLen: maximum length of generated output
        param eos_index: <EOS> token's index
        """
        if maxLen is None:
            maxLen = self.maxLen
        encoder_out, encoder_hidden = self.encoder(source)

        return beamDecoder(self.decoder, encoder_out, encoder_hidden,
                             maxLen, eos_index)

    def tgt2txt(self, tgt):
        return " ".join([self.fields['tgt'].vocab.itos[int(i)] for i in tgt])

    def src2txt(self, src):
        return " ".join([self.fields['src'].vocab.itos[int(i)] for i in src])

    def save(self, file):
        torch.save((self.params, self.state_dict()), file)

    def load(self, file):
        self.params, state_dict = torch.load(file, map_location='cpu')
        self.build()
        self.load_state_dict(state_dict)

class DataFrameDataset(data.Dataset):
    """Class for using pandas DataFrames as a datasource"""
    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        fields = dict(fields)
        self.examples = examples.apply(
            SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

class SeriesExample(data.Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()

        for key, field in fields.items():
            if key not in data:
                raise ValueError(
                    f"Specified key {key} was not found in the input data")
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex

def biload(src_file, tgt_file, linesToLoad=50000, verbose=False):
    src = open(src_file).read().lower().strip().split("\n")
    tgt = open(tgt_file).read().lower().strip().split("\n")
    return list(map(lambda x: (x[0].strip().split(), x[1].strip().split()), zip(src, tgt)))[:linesToLoad]

def bitext2Dataset(src, tgt, srcLex, tgtLex,
                   linesToLoad=50000, maxLen=hp.max_len):
    data = biload(src, tgt, linesToLoad=linesToLoad, verbose=False)
    data = [(f, e) for f, e in data if len(f) <= maxLen and len(e) <= maxLen]
    data = {'src': [f for f, e in data],
            'tgt': [e for f, e in data]}

    df = pd.DataFrame(data, columns=["src", "tgt"])
    dataset = DataFrameDataset(df, [('src', srcLex), ('tgt', tgtLex)])
    return dataset

def loadData(batch_size, device=0,
             trainNum=sys.maxsize, testNum=sys.maxsize):
    def tokenize(x):
        return x.split()

    srcLex = data.Field()
    tgtLex = data.Field(init_token="<SOS>", eos_token="<EOS>")

    train = bitext2Dataset('./data/train.tok.de',
                           './data/train.tok.en', srcLex, tgtLex,
                           linesToLoad=trainNum)
    val = bitext2Dataset('./data/val.tok.de',
                         './data/val.tok.en', srcLex, tgtLex)
    test = bitext2Dataset('./data/input/dev.txt',
                          './data/reference/dev.out', srcLex, tgtLex,
                          linesToLoad=testNum,
                          maxLen=sys.maxsize)

    srcLex.build_vocab(train.src, max_size=hp.lexicon_cap)
    tgtLex.build_vocab(train.tgt, max_size=hp.lexicon_cap)

    train_iter, = data.BucketIterator.splits(
        (train,),
        batch_size=batch_size,
        sort_key=lambda x: len(x.src),
        device=device,
        repeat=False)

    val_iter, = data.BucketIterator.splits(
        (val,),
        batch_size=batch_size,
        device=device,
        repeat=False)

    test_iter = data.Iterator(
        test,
        batch_size=1,
        device=device,
        sort=False,
        sort_within_batch=False,
        shuffle=False,
        repeat=False)

    return train_iter, val_iter, test_iter, srcLex, tgtLex

def loadTestData(srcFile, srcLex, device=0, linesToLoad=sys.maxsize):
    def tokenize(x):
        return x.split()
    test = bitext2Dataset(srcFile,
                          srcFile, srcLex, srcLex, linesToLoad,
                          maxLen=sys.maxsize)
    test_iter = data.Iterator(
        test,
        batch_size=1,
        device=device,
        sort=False,
        sort_within_batch=False,
        shuffle=False,
        repeat=False)
    return test_iter

def loadUnkDict(srcFile, srcLex, linesToLoad=sys.maxsize):
    dataset = bitext2Dataset(srcFile,
                          srcFile, srcLex, srcLex, linesToLoad,
                          maxLen=sys.maxsize)
    unkDict = []
    for i, e in enumerate(dataset.examples):
        s = e.src
        unkDict.append({})
        for ti, token in enumerate(s):
            if dataset.fields['src'].vocab.stoi[token] == 0:
                unkDict[i][ti] = token
    return unkDict

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option(
        "-m", "--model", dest="model", default=os.path.join('data', 'seq2seq_E049.pt'), 
        help="model file")
    optparser.add_option(
        "-i", "--input", dest="input", default=os.path.join('data', 'input', 'dev.txt'),
        help="input file")
    optparser.add_option(
        "-n", "--num", dest="num", default=sys.maxsize, type='int',
        help="num of lines to load")
    (opts, _) = optparser.parse_args()

    model = Seq2Seq(build=False)
    model.load(opts.model)
    model.to(hp.device)
    model.eval()
    # loading test dataset
    test_iter = loadTestData(opts.input, model.fields['src'],
                                device=hp.device, linesToLoad=opts.num)
    unkDict = loadUnkDict(opts.input, model.fields['src'], linesToLoad=opts.num)
    results = translate(model, test_iter, unkDict)
    print("\n".join(results))
