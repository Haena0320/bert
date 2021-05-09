import os
import logging
import sentencepiece as spm
import tqdm
from src.utils import *
from torch.utils.data import DataLoader, Dataset
import re
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
    f = open(model_name+".vocab", encoding="utf-8")
    v = [doc.strip().split("\t") for doc in f]
    word2idx = {w[0]: i for i, w in enumerate(v)}
    # mask token 추가 어캐한담
    word2idx["[MASK]"]= len(word2idx)
    print("vocab size {}".format(len(word2idx)))
    print("mask token id : {}".format(word2idx["[MASK]"]))
    torch.save(word2idx, vocab_path)


def BERTDataloader(config, type, num_workers=10, shuffle=False, drop_last=True):
    bs = config.pretrain.bs
    seq_len = config.pretrain.seq_len
    corpus_path = config.data.bookcorpus[type] # wiki 랑 bookcorpus 합쳐야 함., 이거 수정필요
    vocab = torch.load(config.vocab.bookcorpus) # vocab 합쳐서 30000개임, 이거 수정필요
    dataset = BERTDataset(corpus_path=corpus_path, vocab=vocab, seq_len=seq_len)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return data_loader




class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

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
        lines = [clean_str(line[:-1]) for line in tqdm.tqdm(f, desc="Loading Dataset")]
        self.lines = [lines[i]+lines[i+1] for i in range(len(lines)-1)]
        print(self.lines[0])
        self.corpus_lines = len(self.lines)


    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [cls] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.bos] + t1_random + [self.eos]
        t2 = t2_random + [self.eos]

        t1_label = [self.pad]+t1_label+[self.pad]
        t2_label = t2_label + [self.eos]

        segment_label = ([1 for _ in range(len(t1))]+[2 for _ in range(len(t2))])
        bert_input = (t1+t2)
        bert_label = (t1_label + t2_label)
        if len(seqment_label) < self.seq_len:
            padding = [self.pad for _ in range(self.seq_len - len(bert_input))]
            bert_input.extend(padding)
            bert_label.extend(padding)
            seqment_label.extend(padding)
        else:
            segment_label= segment_label[:self.seq_len]
            bert_input = bert_input[:self.seq_len]
            bert_label = bert_label[:self.seq_len]

        output = {"bert_input":bert_input,
                  "bert_label":bert_label,
                  "segment_label":segment_label,
                  "is_next":is_next_label}

        return {key:torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0) # output of non-target tokens
        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        return self.lines[item][0], self.lines[item][1]


    def get_random_line(self):
        return self.lines[random.randrange(len(self.lines))][1]

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
    return string.strip().lower()
    return string


