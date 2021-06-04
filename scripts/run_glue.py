import sys, os
sys.path.append(os.getcwd())
from torch.utils.tensorboard import SummaryWriter

import argparse
from src.utils import *
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="><")
parser.add_argument("--dataset", type=str, default="SST-2")
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--lr", type=int, default= 3e-4)
parser.add_argument("--use_pretrained", type=int, default=1)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--gpu", type=int, default=None)

args = parser.parse_args()
config = load_config(args.config)
assert args.bs in [8,16,32,64,128]
assert args.lr in [1e-4, 3e-4,2e-5, 5e-5, 3e-5]


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")
print("-------------------------------------------------------------------------------------")
print("training start ! ")
print("current : device {}".format(device))

import sentencepiece as spm
from src.glue_data import *
sp = spm.SentencePieceProcessor()
sp.Load("bpe.model")

# # CoLA dataset
# data_info = config["CoLA"]
# prepro_1("CoLA",data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_1("CoLA", data_info.raw_de, data_info.prepro_de,"dev" ,sp=sp)
# prepro_1("CoLA", data_info.raw_te, data_info.prepro_te,"test", sp=sp)
#
# #MNLI
# data_info = config["MNLI_m"]
# prepro_2("MNLI_m", data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_2("MNLI_m", data_info.raw_de, data_info.prepro_de,"test" ,sp=sp)
#
# data_info = config["MNLI_mm"]
# prepro_2("MNLI_mm", data_info.raw_tr, data_info.prepro_tr, "train",sp=sp)
# prepro_2("MNLI_mm", data_info.raw_de, data_info.prepro_de,"test" ,sp=sp)
#
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

#print("data processing finished !! ")
############################################################################
# model load
from src.model import *
from src.glue_data import *
from src.glue import *
from src.metrics import *

vocab = config.vocab_info.n_token
dimension = config.model.hidden
num_heads = config.model.num_head
num_layers = config.pretrain.num_layers
dim_feedforward = config.model.dim_feedforward
dropout = config.model.d_rate



######### data######################################################################
data = ["MNLI_m","MNLI_mm","MRPC","QNLI", "QQP","RTE","SST_2","WNLI"]
lr_list = [1e-4, 3e-4,2e-5, 5e-5, 3e-5]
bs = [8,16,32,64,128]
total_result = dict()
for target_data in data:
    total_result["dataset"] = target_data
    data_info = config[target_data]
    print(target_data)

    # make log directory
    # log load
    oj = os.path.join
    log_dir = "./log/glue/"
    target_dir = oj(log_dir, target_data)
    tb_dir = oj(target_dir, "tb")
    ckpnt_dir = oj(target_dir, "ckpnt")
    score_dir = oj(target_dir, "score.txt")

    writer = SummaryWriter(tb_dir)
    target_data_result = target_data + "_result"
    total_result[target_data_result] = dict()

    # data load
    for bs_ in bs:
        train_loader = GLUE_Loader(data_info.prepro_tr, "train", bs=bs_)
        if target_data in ["MNLI_m", "MNLI_mm"]:
            test_loader = GLUE_Loader(data_info.prepro_tr, "dev", bs=bs_)  ####
        else:
            test_loader = GLUE_Loader(data_info.prepro_de, "dev", bs=bs_) ####

        print("train: {}".format(len(train_loader)),"test: {}".format(len(test_loader)))

        for lr_ in lr_list:
            bert_model = BertModel(vocab, dimension, num_heads, dim_feedforward, num_layers, dropout, device)
            finetune_dict = bert_model.state_dict()

            ck_path = "log/ckpnt/ckpnt_{}".format(args.use_pretrained)
            checkpoint = torch.load(ck_path, map_location=device)
            pretrain_dict = checkpoint["model_state_dict"]
            # classification layer 제거
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in finetune_dict}
            finetune_dict.update(pretrain_dict)
            bert_model.load_state_dict(finetune_dict)


            glue_task = Classification_Task(config, device, bert_model, data_info.num_labels, data_info.type)
            glue_task.to(device)
            # pretrain
            optimizer = get_optimizer(glue_task, args.optim, lr_)
            trainer = get_trainer(config, args, device,train_loader, sp, writer, "train", data_info.metric)
            tester = get_trainer(config, args, device,test_loader, sp, writer, "test", data_info.metric)

            trainer.init_optimizer(optimizer)

            total_epoch = args.epochs
            #print("{} data experiment start ! | batch length train: {} test: {} | metric : {}".format(target_data, len(train_loader), len(test_loader), data_info.metric))
            #print("total epoch {}".format(total_epoch))
            for epoch in tqdm(range(1, total_epoch+1)):
                trainer.train_epoch(glue_task, epoch, score_dir)
                score, accuracy= tester.train_epoch(glue_task, epoch, score_dir)

                total_result[target_data_result]["bs"] = bs_
                total_result[target_data_result]["lr"] = lr_
                total_result[target_data_result]["score"] = score
                total_result[target_data_result]["accuracy"] = accuracy

torch.save(total_result, "./log/glue/total.pkl")
print("finished ..")
