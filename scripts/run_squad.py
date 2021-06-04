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
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr_rate", type=int, default=5e-5)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--default", type=str, default="default")
parser.add_argument("--use_pretrained", type=int, default=1)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--lr", type=int, default=5e-5)

args = parser.parse_args()
config = load_config(args.default)
assert args.epochs in [2,3,4]
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
sp.Load("bpe.model")

# squad_prepro(data_info.raw_tr, data_info.prepro_tr,  sp)
# squad_prepro(data_info.raw_de, data_info.prepro_de,  sp)
###################################################################################
## model load
from src.model import *
from src.glue_data import *
from src.glue import Question_Answering_Task
from src.metrics import *

vocab = config.vocab_info.n_token
dimension = config.model.hidden
num_heads = config.model.num_head
num_layers = config.pretrain.num_layers
dim_feedforward = config.model.dim_feedforward
dropout = config.model.d_rate

bert_model = BertModel(vocab, dimension, num_heads, dim_feedforward, num_layers, dropout, device)
finetune_dict = bert_model.state_dict()

ck_path = "log/ckpnt/ckpnt_{}".format(args.use_pretrained)
checkpoint = torch.load(ck_path, map_location=device)
pretrain_dict = checkpoint["model_state_dict"]
# classification layer 제거
pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in finetune_dict}
finetune_dict.update(pretrain_dict)

bert_model.load_state_dict(finetune_dict)


######### data######################################################################
target_data = "squad"
data_info = config[target_data]

squad_task = Question_Answering_Task(config, device, bert_model)

squad_task.to(device)
# make log directory
# log load
oj = os.path.join
log_dir = "./log/glue/"
target_dir = oj(log_dir, target_data)
tb_dir = oj(target_dir, "tb")
ckpnt_dir = oj(target_dir, "ckpnt")
score_dir = oj(target_dir, "score.txt")

writer = SummaryWriter(tb_dir)

# data load

train_loader = Squad_Loader(data_info.prepro_tr)
test_loader = Squad_Loader(data_info.prepro_de) ####

print("train: {}".format(len(train_loader)),"test: {}".format(len(test_loader)))

# pretrain
optimizer = get_optimizer(squad_task, args.optim, args.lr)
trainer = get_trainer(config, args, device,train_loader, sp, writer, "train")
tester = get_trainer(config, args, device,test_loader, sp, writer, "test")

trainer.init_optimizer(optimizer)

total_epoch = args.epochs
print("{} data experiment start ! | batch length train: {} test: {}".format(target_data, len(train_loader), len(test_loader)))
print("total epoch {}".format(total_epoch))

for epoch in tqdm(range(1, total_epoch+1)):
    trainer.train_epoch(squad_task, epoch, score_dir)
    tester.train_epoch(squad_task, epoch, score_dir)

print("finished ..")
