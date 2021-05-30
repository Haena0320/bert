import sys, os
sys.path.append(os.getcwd())
import argparse
import torch
from tqdm import tqdm
from src.utils import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="")

parser.add_argument("--config", type=str, default="default")
parser.add_argument("--total_steps", type=int, default="1000000")
parser.add_argument("--dataset", type=str, default="bookcorpus")
parser.add_argument("--model", type=str, default="base")
parser.add_argument("--eval_steps", type=int, default=50000)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--use_pretrained", type=int, default=0)
parser.add_argument("--total_step", type=int, default=1000000)
parser.add_argument("--epochs", type=int, default=100)

args = parser.parse_args()
config = load_config(args.config)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else  "cpu")


## log save file
oj = os.path.join

log = "./log/"
ckpnt_loc = oj(log, "ckpnt/")
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

# #############################################train mode ###############################################################
#data load
import glob
file_list = glob.glob("./data/raw/bookcorpus_s/*")
print("data file num : {}".format(len(file_list)))

#model load
model = BERT_PretrainModel(config, args, device)

#trainer load
trainer = train.get_trainer(config,args,device, file_list, sp, writer, "train")

if args.use_pretrained:
    ck_path = oj(ck_loc, "/ckpnt_{}".format(args.use_pretrained))
    #ck_path = "/data/user15/workspace/BERT/log/ckpntckpnt_1"
    checkpoint = torch.load(ck_path, map_location=device)
    print(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = train.get_optimizer(model, args.optim)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = train.get_lr_scheduler(optimizer, config)
    scheduler._step = checkpoint['lr_step']

    trainer.init_optimizer(optimizer)
    trainer.init_scheduler(scheduler)

    total_epoch = checkpoint["epoch"]

    model.train()

else:
    optimizer = train.get_optimizer(model, args.optim)
    scheduler = train.get_lr_scheduler(optimizer, config)

    trainer.init_optimizer(optimizer)
    trainer.init_scheduler(scheduler)

    total_epoch = args.epochs
    print("total epoch {}".format(total_epoch))

for epoch in tqdm.tqdm(range(1, total_epoch+1)):
    trainer.train_epoch(model, epoch, save_path=ckpnt_loc)
    if trainer.global_step > args.total_steps:
        break
print('finished...')
