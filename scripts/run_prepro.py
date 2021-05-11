import torch
from src.prepro import *
from src.utils import *
import logging
import os, sys
sys.path.append(os.getcwd())

config = load_config("default")
type = ["train", "test", "valid"]
#############################################################pretraining################################################
##################################bookcorpus
vocab = make_vocab(config.data.bookcorpus.large, config.vocab.bookcorpus, config.vocab_info.n_token, config.vocab_info.model_name, config.vocab_info.model_type)
path = {"train":config.bookcorpus_raw.train, "test":config.bookcorpus_raw.test, "valid":config.bookcorpus_raw.valid}




##################################wiki
vocab = make_vocab(config.data.wiki.large,config.vocab.wiki, config.vocab_info.n_token, config.vocab_info.model_name, config.vocab_info.model_type)
path = {"train":config.wiki_raw.train, "test":config.wiki_raw.test, "valid":config.wiki_raw.valid}

#############################################################fine-tuning################################################






