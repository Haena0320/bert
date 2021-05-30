import json
from torch.utils.data import DataLoader, Dataset
import logging
import torch


def squad_prepro(raw_path, save_path,sp, seq_len=128):
    pad = 0
    bos = 1
    eos = 2
    unk = 3
    f = open(raw_path, encoding="utf-8")
    squad = json.load(f)
    
    data = dict()
    data['data'] = list()
    for article in squad["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            context = sp.EncodeAsIds(context)

            for qa in paragraph["qas"]:

                question = sp.EncodeAsIds(qa["question"].strip())
                answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                if len(answer_starts) == 0:
                    continue
                else:
                    answer_starts = answer_starts[0]
                answers = [sp.EncodeAsIds(answer["text"].strip()) for answer in qa["answers"]][0]


                _input = [bos]+question + [eos]+ context
                _segment = [pad]+[1]*len(question)+[pad]+[2]*len(context)
                _input += [pad]*(seq_len-len(_input))
                _segment += [pad]*(seq_len-len(_segment))

                _label = [answer_starts]+[answer_starts+len(answers)-1] # start, end token

                assert len(_input) > len(_label)
                assert len(_input) == len(_segment)

                """
                item  = {
                    "context": context,
                    "question": question,
                    "answers": {
                        "answer_start": answer_starts,
                        "text": answers,
                    }
                }
                """
                item = {"input":_input,"segment":_segment ,"label":_label}
                data["data"].append(item)

    torch.save(data, save_path)
    print("make dataset : {}".format(len(data['data'])))
    print('finished !! ')
    return None

    
class Squad_Dataset(Dataset):
    def __init__(self, filepath):
        logging.info("generating examples from = %s", filepath)
        f = open(filepath, encoding="utf-8")
        self.squad = json.load(f)

    def __len__(self):
        return len(self.squad)

    def _getitem__(self, item):
        inputs = self.squad["data"][item]
        inputs = {k:torch.LongTensor(v) for k, v in inputs.items()}
        return inputs

def Squad_Loader(corpus_path, bs=32, num_workers=3, shuffle=True,drop_last=True):
    dataset = Squad_Dataset(corpus_path)
    data_loader = DataLoader(dataset, batch_size=bs, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)
    return data_loader

def get_optimizer(model, args_optim, lr):
    if args_optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-09)

def get_trainer(config, args, device, data_loader,  sp, writer, type):
    return Trainer(config, args, device,data_loader, sp, writer, type)

class Trainer:
    def __init__(self, config, args, device,data_loader ,sp, writer, type):
        self.config = config
        self.args = args
        self.device = device
        self.writer = writer
        self.type = type
        self.data_loader = data_loader
        self.sp = sp

        self.get_loss = nn.CrossEntropyLoss(ignore_index=0)

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer

    def init_scheduler(self, scheduler):
        self.scheduler = scheduler

    def log_writer(self, log, step):
        if self.type =="train":
            self.writer.add_scalar('train/loss', log, self.global_step)
            self.writer.add_scalar('train/lr', lr, self.global_step)

        else:
            self.writer.add_scalar("valid/loss", log, step)

    def train_epoch(self, model, epoch, save_path=None):
        if self.type =="train":
            model.train()

        else:
            model.eval()
        model.to(self.device)
        loss_save = list()
        total_accuracy = 0 
        
        for data in tqdm(self.data_doader):
            data = {k:v.to(self.device) for k, v in data.items()}
            squad_output = model(data)
            loss = get_loss(squad_output, data["label"])
            pass



import sys, os
sys.path.append(os.getcwd())
from src.model import *

class Squad_Task(nn.Module):
    def __init__(self, config, device):
        super(Squad_Task,self).__init__()
        vocab = config.vocab_info.n_token
        dimension = config.model.hidden
        num_heads = config.model.num_head
        num_layers = config.pretrain.num_layer
        dim_feedforward = config.model.dim_feedforward
        dropout = config.model.d_rate

        self.bert_model = BertModel(vocab, dimension, num_heads, dim_feedforward, num_layers, dropout, device)



    def forward(self, data):
        output = self.model(data["input"], data["segment"])

def get_loss(output, label):
    """
    :param output: (bs, seq, dimension)
    :param label: [start token, end token], list() , len(label) == 2
    :return: loss
    """

            










        
        



