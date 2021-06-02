import json
from torch.utils.data import DataLoader, Dataset
import logging
import torch
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from src.metrics import compute_f1
from tqdm import tqdm

def squad_prepro(raw_path, save_path, sp, seq_len=128):
    pad = 0
    bos = 1
    eos = 2
    unk = 3
    f = open(raw_path, encoding="utf-8")
    squad = json.load(f)

    data = dict()
    data['data'] = list()
    no_answer_cnt = 0
    for article in tqdm(squad["data"], desc="make data"):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            context = sp.EncodeAsIds(context)

            for qa in paragraph["qas"]:

                question = sp.EncodeAsIds(qa["question"].strip())
                answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                if len(answer_starts) == 0: # squad v2.2
                    _label = [[bos]]
                    no_answer_cnt += 1

                else: # squad v1.1

                    _label = list(set([answer["text"].strip() for answer in qa["answers"]])) # answer token list [1,2,3,5]
                    _label = [sp.EncodeAsIds(l) for l in _label]

                _input = [bos] + question + [eos] + context
                _segment = [1] + [1] * len(question) + [1] + [2] * len(context)
                _input += [pad] * (seq_len - len(_input))
                _segment += [pad] * (seq_len - len(_segment))

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
                item = {"input": _input, "segment": _segment, "label": _label}
                data["data"].append(item)

    torch.save(data, save_path)
    print("sample dataset")
    print("no answer question : {}".format(no_answer_cnt))
    print("total data pair : {}".format(len(data["data"])))
    print('finished !! ')
    return None

"""
sample
{'input': [1, 3, 815, 41, 326, 13, 6, 3365, 3, 2, 3, 162, 10, 3, 568, 17424, 354, 115, 6, 3, 6, 3, 41, 6, 815, 24592, 80, 4366, 5104, 30, 58, 3, 13, 3, 3435, 7, 74, 3, 33, 2964, 354, 3429, 13, 3, 6, 3, 12011, 3, 3401, 75, 3, 41, 22,
 2631, 22, 3, 34, 3, 64, 14152, 7, 10, 815, 13, 3, 41, 37, 10, 326, 13, 6, 3365, 3, 9, 41, 4341, 3, 705, 18, 113, 2522, 444, 32, 108, 6512, 31, 14096, 6110, 3, 3263, 3, 11807, 562, 3, 27506, 9470, 17196, 9, 1910, 24804, 3, 15204, 76
92, 13, 815, 5530, 6, 3, 170, 41, 8987, 7, 3, 9, 6, 3, 170, 41, 8987, 7, 3, 0, 0, 0, 0, 0, 0, 0], 'segment': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
2, 2, 2, 0, 0, 0, 0, 0, 0, 0], 'label': [[1]]}"""

    
class Squad_Dataset(Dataset):
    def __init__(self, filepath):
        logging.info("generating examples from = %s", filepath)
        self.squad = torch.load(filepath)["data"]

    def __len__(self):
        return len(self.squad)

    def __getitem__(self, item):
        inputs = self.squad[item]
        return inputs


def padd_fn(samples):
    def padd(samples):
        ln = [len(l) for l in samples]
        max_ln = max(ln)
        data = torch.zeros(len(samples), max_ln).to(torch.long)
        for i, sample in enumerate(samples):
            data[i, :ln[i]] = torch.LongTensor(sample)
        return LongTensor(data)
    label_ = [sample['label'] for sample in samples]
    label_ = padd(label_)

    return {"input": torch.LongTensor(samples["input"]).contiguous(),
            "segment": torch.LongTensor(samples["segment"]).contiguous(),
            "label": torch.LongTensor(label_).contiguous()}


def Squad_Loader(corpus_path, bs=32, num_workers=3, shuffle=True,drop_last=True):
    dataset = Squad_Dataset(corpus_path)
    data_loader = DataLoader(dataset, batch_size=bs, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=padd_fn)
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

        self.step = 0
        self.clip = 5

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
        total_loss  = list()
        total_accuracy = list()
        total_f1 = list()
        
        for data in tqdm(self.data_loader):
            print(data)
            data = {k:v.to(self.device) for k, v in data.items()}
            squad_output = model(data)
            loss, accuracy = get_loss(squad_output, data["label"])
            
            f1 = compute_f1(squad_output,data["label"],data['answers'])
            
            if self.type =="train":
                
                self.optim_process(model, loss)
                self.step += 1
                self.writer.add_scalar("train/accuracy", accuracy.data, self.step)
                self.writer.add_scalar("train/f1", f1.data, self.step)

                if self.step  % self.ckpnt_step ==0:
                    torch.save({"epoch":epoch,
                                "model_state_dict":model.state_dict(),
                                "optimizer_state_dict":self.optimizer.state_dict()},
                               save_path+"cknpt_{}".format(epoch))
            else:
                total_loss.append(loss.item())
                total_accuracy.append(accuracy.item())
                total_f1.append(f1.item())
                
        if self.type != "train":
            self.writer.add_scalar("test/loss", np.mean(total_loss), self.step)
            self.writer.add_scalar("test/accur", np.mean(total_accuracy), self.step)
            self.writer.add_scalar("test/f1", np.mean(total_f1), self.step)
            

    def optim_process(self, model, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.log_writer(loss.data, self.step)


import sys, os
sys.path.append(os.getcwd())
from src.model import *

class Squad_Task(nn.Module):
    def __init__(self, config, device, model):
        super(Squad_Task,self).__init__()

        #self.bert_model = BertModel(vocab, dimension, num_heads, dim_feedforward, num_layers, dropout, device)
        self.bert_model = model
        self.sep = 2

    def forward(self, data):
        hidden = self.bert_model(data["input"], data["segment"]) #(bs, seq, dimension)
        labels = data['label'] # (bs, 2)
        bs, seq, _ = hidden.size()
        final_pred = torch.zeros((bs,2))
        for idx in range(bs):
            target = hidden[idx, :,:]
            start = labels[idx][0]
            end = labels[idx][1]
            for s in range(seq):
                if target[s] ==2:
                    target = target[s+1:]

                # target : predict token
                _start = F.softmax(start*target)
                _end = F.softmax(end*target)
                pred = _start.unsqueeze(0).repeat(len(target),1)+_end.unqueeze(1).repeat(1,len(target))
                pred = torch.tril(pred)
                i = torch.argmax(pred)//len(target)
                p = torch.argmax(pred)%len(target)
                final_pred[idx] = torch.cat([i.unsqueeze(0),p.unsqueeze(0)])

        return final_pred

def get_loss(output, label):
    """
    :param output: (bs, seq, dimension)
    :param label: [start token, end token], list() , len(label) == 2
    :return: loss
    """
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, label)
    bs, span = label.size()
    accuracy = sum(output.eq(label).float())*100/(bs*span)
    return loss, accuracy

            










        
        



