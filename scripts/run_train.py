import sys, os
sys.path.append(os.getcwd())
import argparse
import torch
from tqdm import tqdm
from src.utils import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--total_steps", type=int, default="90000")
parser.add_argument("--dataset", type=str, default="bookcorpus")
parser.add_argument("--model", type=str, default="base")
parser.add_argument("--eval_steps", type=int, default=50000)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--optim", type=str, default="adam")

args = parser.parse_args()
config = load_config(args.config)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.device) if use_cuda and args.device is not None else  "cpu")

## log save file
oj = os.path.join

log = "./log/"
ckpnt_loc = oj(log, "ckpnt")
loss_loc = oj(log, "loss")
eval_loc = oj(log, "eval_loss")


if not os.path.exists(ckpnt_loc):
    #os.mkdir(log)
    os.mkdir(ckpnt_loc)
   # os.mkdir(loss_loc)
    os.mkdir(eval_loc)

writer = SummaryWriter(loss_loc)

from src.model import BERT_PretrainModel
import src.train as train
from src.prepro import *
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("word-piece-encoding.model")

#############################################debug######################################################################
# # data load
# debug_loader = BERTDataloader(config, "debug", sp)
# model = BERT_PretrainModel(config, args, device)
#
# # trainer load
# debuger = train.get_trainer(config, args, device, debug_loader, writer, "train")
#
# # optimizer
# optimizer = train.get_optimizer(model, args.optim)
# scheduler = train.get_lr_scheduler(optimizer, config)
#
# debuger.init_optimizer(optimizer)
# debuger.init_scheduler(scheduler)
#
# # train + eval
# epochs = args.total_steps // len(debug_loader)
# print("-------------------------------------------------Train Epochs  {}------------------------------------------------".format(epochs))
# for epoch in range(epochs):
#     debuger.train_epoch(model,epoch)
#
# print("train finished..")
#

##############################################train mode ###############################################################
#data load

train_loader = BERTDataloader(config,"train", sp)
test_loader = BERTDataloader(config, "test", sp)
valid_loader = BERTDataloader(config, "valid", sp)

print("Iteration :: train {} | test {} | valid {}".format(len(train_loader), len(test_loader), len(valid_loader)))

#model load
model = BERT_PretrainModel(config, args, device)

#trainer load
trainer = train.get_trainer(config,args,device,train_loader, writer, "train")
tester = train.get_trainer(config, args, device, test_loader, writer, "test")
valider = train.get_trainer(config, args, device, valid_loader, writer, "valid")

# optimizer
optimizer = train.get_optimizer(model, args.optim)
scheduler = train.get_lr_scheduler(optimizer, config)

trainer.init_optimizer(optimizer)
trainer.init_scheduler(scheduler)

#train + eval
epochs = max(args.total_steps // len(train_loader), 1)
print("-------------------------------------------------Train Epochs  {}------------------------------------------------".format(epochs))
for epoch in range(epochs):
   trainer.train_epoch(model,epoch)
   valider.train_epoch(model, epoch)
   tester.train_epoch(model, epoch)

print("train finished..")
