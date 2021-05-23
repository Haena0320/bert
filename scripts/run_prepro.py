import os, sys
sys.path.append(os.getcwd())
import torch
from src.prepro import *
from src.data import *
from src.utils import *
import logging


config = load_config("default")
types = ['train', 'test', "valid"]
#############################################################pretraining################################################
##################################bookcorpus
# vocab = make_vocab(config.data.bookcorpus.large, config.vocab.bookcorpus, config.vocab_info.n_token, config.vocab_info.model_name, config.vocab_info.model_type)
path = [config.data.bookcorpus.test, config.data.bookcorpus.valid]
save_path = [config.prepro_data.bookcorpus.test, config.prepro_data.bookcorpus.valid]

## make training data
vocab = torch.load(config.vocab.bookcorpus)
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("word-piece-encoding.model")
seq_len = config.pretrain.seq_len
for p, s_p in list(zip(path, save_path)):
    make_data = Make_BERTDataset(p, vocab, seq_len, sp, encoding="utf-8", corpus_lines=None, on_memory=True)
    make_data.data_prepro(s_p)


#########################################################################################save ##########################

