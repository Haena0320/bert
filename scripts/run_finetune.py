import sys, os
sys.path.append(os.getcwd())
from datasets import load_dataset
import torch
import argparse
from src.utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="?_?")
parser.add_argument("--dataset", type=str, default="squad")
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--lr_rate", type=int, default=5e-5)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--default", type=str, default="default")
parser.add_argument("--use_pretrained", type=int, default=0)
parser.add_argument("--optim", type=str, default="adam")

args = parser.parse_args()
config = load_config(args.default)
assert args.epoch in [2,3,4]
assert args.dataset in ["squad", "squad_v2","glue", "swag"]
assert args.lr_rate in [5e-5, 4e-5, 3e-5, 2e-5]

# prepro
import sentencepiece as spm
from src.squad import *

sp = spm.SentencePieceProcessor()
sp.Load("word-piece-encoding.model")

data_info = config[args.dataset]

squad_prepro(data_info.raw_tr, data_info.prepro_tr,  sp)
squad_prepro(data_info.raw_de, data_info.prepro_de,  sp)

# data load
from src.squad import *
train_loader = Squad_Loader(data_info.prepro_tr)
dev_loader = Squad_Loader(data_info.prepro_de)

# model load
from src.model import BERT_PretrainModel
model = BERT_PretrainModel
ck_path = oj(ck_loc, "/ckpnt_ckpnt_{}".format(args.use_pretrained))
checkpoint = torch.load(ck_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])

# pretrain
squad_task = Squad_Task(model)
optimizer = get_optimizer(squad_task, args.optim, args.lr)
trainer = get_trainer(config, args, device, sp, writer, "train")
valider = get_trainer(config, args, device, sp, writer, "dev")

trainer.init_optimizer(optimizer)
model.train()

total_epoch = args.epoch
print("total epoch {}".format(total_epoch))
for epoch in tqdm.tqdm(range(1, total_epoch+1)):
    trainer.train_epoch(squad_task, epoch)
    valider.train_epoch(squad_task, epoch)
    
print("finished ..")
# eval





