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
parser.add_argument("--total_steps", type=int, default="1000000")
parser.add_argument("--dataset", type=str, default="bookcorpus")
parser.add_argument("--model", type=str, default="base")
parser.add_argument("--eval_steps", type=int, default=50000)

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


if not os.path.exists(log):
    os.mkdir(log)
    os.mkdir(ckpnt_loc)
    os.mkdir(loss_loc)
    os.mkdir(eval_loc)

writer = SummaryWriter(loss_loc)

from src.model import BERT_PretrainModel
import src.train as train
from src.prepro import *

# data load
debug_loader = BERTDataloader(config, "train")
# train_loader = BERTDataloader(config,"train")
# test_loader = BERTDataloader(config, "test")
# valid_loader = BERTDataloader(config, "valid")
#
# print("Iteration :: train {} | test {} | valid {}".format(len(train_loader), len(test_loader), len(valid_loader)))

# model load
model = BERT_PretrainModel()

# trainer load
trainer = train.get_trainer(config,args,device,train_loader, writer, "train")
# tester = train.get_trainer(config, args, device, test_loader, writer, "test")
# valider = train.get_trainer(config, args, device, valid_loader, writer, "valid")

# train + eval
epochs = args.total_steps // len(train_loader)
for epoch in range(epochs):
    trainer.train_epoch(model,epoch)

print("train finished..")
