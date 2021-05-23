import sentencepiece as spm
from tqdm import tqdm
import re
import random
import numpy as np
import logging
import torch
import json

class Make_BERTDataset:
    def __init__(self, corpus_path, vocab, seq_len, sp, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.sp = sp
        self.vocab_size = len(vocab)

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.pad = 0
        self.bos = 1
        self.eos = 2
        self.unk = 3
        self.mask = self.vocab["[MASK]"]
        f = open(corpus_path, 'r', encoding="utf-8")
        lines = [[clean_str(line[:-1])] for line in tqdm(f, desc="Loading Dataset")]
        self.lines = [lines[i] + lines[i + 1] for i in range(len(lines) - 1)]
        f.close()
        self.corpus_lines = len(self.lines)

    def data_prepro(self, save_path):

        data_dummies = dict()
        data_dummies["data"] = list()
        for idx in tqdm(range(self.corpus_lines)):
            data = self.item(idx)
            data_dummies["data"].append(data)
        with open(save_path, 'w') as outfile:
            json.dump(data_dummies, outfile)
        logging.info("save data ..")

    def item(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [cls] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.bos] + t1_random
        t2 = [self.eos] + t2_random

        t1_label = [self.pad] + t1_label
        t2_label = [self.pad] + t2_label

        segment_input = [0] * len(t1) + [1] * len(t2)
        bert_input = t1 + t2
        bert_label = t1_label + t2_label

        if len(segment_input) < self.seq_len:
            padding = [self.pad] * (self.seq_len - len(bert_input))  # for _ in range(self.seq_len - len(bert_input))]
            bert_input.extend(padding)
            bert_label.extend(padding)
            segment_input.extend(padding)
        else:
            segment_input = segment_input[:self.seq_len]
            bert_input = bert_input[:self.seq_len]
            bert_label = bert_label[:self.seq_len]

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_input": segment_input,
                  "is_next": is_next_label}

        assert len(bert_input) == len(segment_input)
        assert is_next_label in [0, 1]

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

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            t2 = self.get_random_line()
            return t1, t2, 0

    def get_corpus_line(self, item):
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        return self.lines[random.randint(0, self.corpus_lines-1)][0]


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
    #return string.strip().lower()
    return string
