import os, sys

sys.path.append(os.getcwd())
import logging
import sentencepiece as spm
import tqdm
from src.utils import *
from torch.utils.data import DataLoader, Dataset
import re
import random
import numpy as np


# bookcorpus data
# URL_book = "https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2"
# wikitext2
# URL_wiki = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
# train(36718), valid(3760), test(4358)

def make_vocab(input_file, vocab_path, vocab_size, model_name, model_type):
    pad = 0
    bos = 1
    eos = 2
    unk = 3

    input_argument = "--input=%s --model_prefix=%s --vocab_size=%s --pad_id=%s --bos_id=%s --eos_id=%s --unk_id=%s --model_type=%s"
    cmd = input_argument % (input_file, model_name, vocab_size, pad, bos, eos, unk, model_type)

    spm.SentencePieceTrainer.Train(cmd)
    logging.info("model, vocab finished ! ")
    f = open(model_name + ".vocab", encoding="utf-8")
    v = [doc.strip().split("\t") for doc in f]
    word2idx = {w[0]: i for i, w in enumerate(v)}
    # mask token 추가 어캐한담
    word2idx["[MASK]"] = len(word2idx)
    print("vocab size {}".format(len(word2idx)))
    print("mask token id : {}".format(word2idx["[MASK]"]))
    torch.save(word2idx, vocab_path)


def BERTDataloader(config, type, sp, num_workers=10, shuffle=True, drop_last=True):
    bs = config.pretrain.bs
    seq_len = config.pretrain.seq_len
    corpus_path = config.data.bookcorpus[type]  # wiki 랑 bookcorpus 합쳐야 함., 이거 수정필요
    vocab = torch.load(config.vocab.bookcorpus)
    dataset = BERTDataset(corpus_path=corpus_path, vocab=vocab, sp=sp)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
                             collate_fn=make_padd)
    return data_loader


def make_padd(samples):
    bert_input = [sample["bert_input"] for sample in samples]
    bert_label = [sample["bert_label"] for sample in samples]
    segment_input = [sample["segment_input"] for sample in samples]

    def padd(samples):
        length = [len(i) for i in samples]
        max_len = 128
        batch = torch.zeros((len(length), max_len)).to(torch.long)
        for idx, sample in enumerate(samples):
            if length[idx] < max_len:
                batch[idx, :length[idx]] = torch.LongTensor(sample)
            else:
                batch[idx, :max_len] = torch.LongTensor(sample[:max_len])
                batch[idx, max_length - 1] = troch.LongTensor([2])
        return torch.LongTensor(batch)

    bert_input = padd(bert_input)
    bert_label = padd(bert_label)
    segment_input = padd(segment_input)

    return {"bert_input": bert_input.contiguous(), "bert_label": bert_label.contiguous(),
            "segment_input": segment_input.contiguous()}


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, sp, encoding="utf-8"):
        self.vocab = vocab
        self.sp = sp
        self.vocab_size = len(vocab)

        self.pad = 0
        self.bos = 1
        self.eos = 2

        self.mask = self.vocab["[MASK]"]
        f = open(corpus_path, 'r', encoding=encoding)
        self.lines = [[clean_str(line[:-1])] for line in tqdm.tqdm(f, desc="Loading Dataset")]
        f.close()
        self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t = self.lines(item)
        t_random, t_label = self.random_word(t)
        t_random.insert(0, self.bos)
        t_label.insert(0, self.pad)
        segment_input = [self.pad] * len(t)

        output = {"bert_input": t_random,
                  "bert_label": t_label,
                  "segment_input": segment_input}
        return output

    def random_word(self, sentence):
        tokens = self.sp.EncodeAsIds(sentence)
        token_probs = np.random.uniform(0, 1, len(tokens)).tolist()
        probs = list(zip(tokens, token_probs))
        output_label = [0] * len(tokens)
        cnt = 0

        for token, p in probs:
            if p < 0.15:
                # 80% randomly change token to mask token
                if p < 0.12:
                    tokens[cnt] = self.mask
                # 10% randomly change token to random token
                elif p < 0.135:
                    tokens[cnt] = np.random.randint(0, self.vocab_size)
                # 10% randomly change token to current token
                else:
                    pass
                output_label[cnt] = token
            cnt += 1
        assert len(tokens) == len(output_label)
        return tokens, output_label


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string



