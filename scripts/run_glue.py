import sys, os
sys.path.append(os.getcwd())
from torch.utils.tensorboard import SummaryWriter

import argparse
from src.utils import *
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="><")
parser.add_argument("--dataset", type=str, default="SST-2")
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr_rate", type=int, default=5e-5)


args = parser.parse_args()
config = load_config(args.config)

import sentencepiece as spm
from src.glue_data import *
sp = spm.SentencePieceProcessor()
sp.Load("word-piece-encoding.model")

# # CoLA dataset
# data_info = config["CoLA"]
# prepro_1("CoLA",data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_1("CoLA", data_info.raw_de, data_info.prepro_de,"dev" ,sp=sp)
# prepro_1("CoLA", data_info.raw_te, data_info.prepro_te,"test", sp=sp)

# MNLI
# data_info = config["MNLI_m"]
# prepro_2("MNLI_m", data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_2("MNLI_m", data_info.raw_de, data_info.prepro_de,"test" ,sp=sp)
#
# data_info = config["MNLI_mm"]
# prepro_2("MNLI_mm", data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_2("MNLI_mm", data_info.raw_de, data_info.prepro_de,"test" ,sp=sp)

# # MRPC
# data_info = config["MRPC"]
# prepro_2("MRPC",data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_2("MRPC", data_info.raw_de, data_info.prepro_de,"dev" ,sp=sp)
# prepro_2("MRPC", data_info.raw_te, data_info.prepro_te,"test", sp=sp)
#
# #QNLI
# data_info = config["QNLI"]
# prepro_2("QNLI",data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_2("QNLI", data_info.raw_de, data_info.prepro_de,"dev" ,sp=sp)
# prepro_2("QNLI", data_info.raw_te, data_info.prepro_te,"test", sp=sp)
#
# #QQP
# data_info = config["QQP"]
# prepro_2("QQP",data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_2("QQP", data_info.raw_de, data_info.prepro_de,"dev" ,sp=sp)
# prepro_2("QQP", data_info.raw_te, data_info.prepro_te,"test", sp=sp)
#
# #RTE
# data_info = config["RTE"]
# prepro_2("RTE",data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_2("RTE", data_info.raw_de, data_info.prepro_de,"dev",sp=sp)
# prepro_2("RTE", data_info.raw_te, data_info.prepro_te,"test", sp=sp)
#
# #SST_2
# data_info = config["SST_2"]
# prepro_1("SST_2",data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_1("SST_2", data_info.raw_de, data_info.prepro_de,"dev",sp=sp)
# prepro_1("SST_2", data_info.raw_te, data_info.prepro_te,"test", sp=sp)
#
# #STS_B
# data_info = config["STS_B"]
# prepro_2("STS_B",data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_2("STS_B", data_info.raw_de, data_info.prepro_de,"dev",sp=sp)
# prepro_2("STS_B", data_info.raw_te, data_info.prepro_te,"test", sp=sp)
#
# #WNLI
# data_info = config["WNLI"]
# prepro_2("WNLI",data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_2("WNLI", data_info.raw_de, data_info.prepro_de,"dev",sp=sp)
# prepro_2("WNLI", data_info.raw_te, data_info.prepro_te,"test", sp=sp)

print("data processing finished !! ")
############################################################################
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

trainer.init_optimizer(optimizer)


######### data
target_data = "CoLA"
# make log directory
# log load
oj = os.path.join()
log_dir = "./log/glue/"
target_dir = oj(log_dir, target_data)
tb_dir = oj(target_dir, "tb")
ckpnt_dir = oj(target_dir, "ckpnt")

writer = Summarywriter(tb_dir)

# data load
from src.glue_data import *
data_info = config[target_data]
train_loader = GLUE_Loader(data_info.prepro_tr)
test_loader = GLUE_Loader(data_info.prepro_te)
print("train: {}".format(len(train_loader)),"test: {}".format(len(test_loader)))

# pretrain
from src.glue import *
glue_task = Classification_Task(config, device, model, data_info.num_labels, data_info.type, data_info.metric)
optimizer = get_optimizer(glue_task, args.optim, args.lr)
trainer = get_trainer(config, args, device,train_loader, sp, writer, "train")
tester = get_trainer(config, args, device,dev_loader, sp, writer, "test")

total_epoch = args.epochs
print("total epoch {}".format(total_epoch))

for epoch in tqdm(range(1, total_epoch+1)):
    trainer.train_epoch(glue_task, epoch)
    tester.train_epoch(glue_task, epoch)

print("finished ..")
