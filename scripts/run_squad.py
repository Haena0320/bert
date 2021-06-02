import sys, os
sys.path.append(os.getcwd())
from datasets import load_dataset
import torch
import argparse
from src.utils import load_config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="?_?")
parser.add_argument("--dataset", type=str, default="squad")
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--lr_rate", type=int, default=5e-5)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--default", type=str, default="default")
parser.add_argument("--use_pretrained", type=int, default=1)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--lr", type=int, default=5e-5)

args = parser.parse_args()
config = load_config(args.default)
assert args.epoch in [2,3,4]
assert args.dataset in ["squad", "squad_v2","glue", "swag"]
assert args.lr_rate in [5e-5, 4e-5, 3e-5, 2e-5]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")
print("-------------------------------------------------------------------------------------")
print("training start ! ")
print("current : device {}".format(device))

from src.squad import *
data_info = config[args.dataset]
## prepro ######################################################################
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("word-piece-encoding.model")

# squad_prepro(data_info.raw_tr, data_info.prepro_tr,  sp)
# squad_prepro(data_info.raw_de, data_info.prepro_de,  sp)
###################################################################################
# log dir
log_dir = "./log/"
oj = os.path.join
dir = oj(log_dir, args.dataset)
tb_dir = oj(dir, "tb")

if not os.path.exists(dir):
    os.mkdir(dir)
    os.mkdir(tb_dir)

writer = SummaryWriter(tb_dir)

# data load
from src.squad import *
train_loader = Squad_Loader(data_info.prepro_tr)
dev_loader = Squad_Loader(data_info.prepro_de)
print("train: {}".format(len(train_loader)),"test: {}".format(len(dev_loader)))

# model load
from src.model import *
model = BERT_PretrainModel(config,args,device)

vocab = config.vocab_info.n_token
dimension = config.model.hidden
num_heads = config.model.num_head
num_layers = config.pretrain.num_layers
dim_feedforward = config.model.dim_feedforward
dropout = config.model.d_rate
bert_model = BertModel(vocab, dimension, num_heads, dim_feedforward, num_layers, dropout, device)

ck_path = "log/ckpnt/ckpnt_{}".format(args.use_pretrained)
checkpoint = torch.load(ck_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

# pretrain
squad_task = Squad_Task(config, device, model)
optimizer = get_optimizer(squad_task, args.optim, args.lr)
trainer = get_trainer(config, args, device,train_loader, sp, writer, "train")
valider = get_trainer(config, args, device,dev_loader, sp, writer, "dev")

trainer.init_optimizer(optimizer)
model.train()

total_epoch = args.epoch
print("total epoch {}".format(total_epoch))
for epoch in tqdm(range(1, total_epoch+1)):
    trainer.train_epoch(squad_task, epoch)
    valider.train_epoch(squad_task, epoch)

print("finished ..")
# eval
##############################################################################

