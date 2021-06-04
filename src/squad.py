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
                    _label = [bos]
                    no_answer_cnt += 1

                else: # squad v1.1
                    _label = qa["answers"][0]["text"]
                    _label = sp.EncodeAsIds(_label)
                    if len(_label) == 1:
                        _label = _label + _label
                        assert len(_label) == 2
                _input = [bos] + question + [eos] + context
                _segment = [1] + [1] * len(question) + [1] + [2] * len(context)
                _start = [1 if _input[i] == _label[0] else 0 for i in range(len(_input))]
                _end = [1 if _input[i] == _label[1] else 0 for i in range(len(_input))]
                
                _input += [pad] * (seq_len - len(_input))
                _start += [pad]*(seq_len - len(_start))
                _end += [pad]*(seq_len - len(_end))
                _segment += [pad] * (seq_len - len(_segment))


                assert len(_input) == len(_segment)
                assert len(_input) == len(_start)
                assert len(_input) == len(_end)

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
                item = {"inputs": _input, "segments": _segment, "starts": _start, "ends":_end}
                data["data"].append(item)

    torch.save(data, save_path)
    print("sample dataset")
    print("no answer question : {}".format(no_answer_cnt))
    print("total data pair : {}".format(len(data["data"])))
    print('finished !! ')
    return None



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
        max_ln = 128
        data = torch.zeros(len(samples), max_ln).to(torch.long)
        for i, sample in enumerate(samples):
            if ln[i]> max_ln:
                data[i,:] = torch.LongTensor(sample[:max_ln])
            else:
                data[i, :ln[i]] = torch.LongTensor(sample)
        return torch.LongTensor(data)
    def label_padd(samples):
        ln = [len(l) for l in samples]
        max_ln = 128
        data = torch.zeros(len(samples), max_ln).to(torch.float)
        for i, sample in enumerate(samples):
            if ln[i]> max_ln:
                data[i,:] = torch.FloatTensor(sample[:max_ln])
            else:
                data[i, :ln[i]] = torch.FloatTensor(sample)
        return torch.FloatTensor(data)

    inputs = [sample['inputs'] for sample in samples]
    segments = [sample["segments"] for sample in samples]
    starts = [sample["starts"] for sample in samples]
    ends = [sample["ends"] for sample in samples]

    inputs = padd(inputs)
    segments = padd(segments)
    starts = padd(starts)
    ends = padd(ends)

    return {"inputs": torch.LongTensor(inputs).contiguous(),
            "segments": torch.LongTensor(segments).contiguous(),
            "starts": torch.LongTensor(starts).contiguous(),
            "ends":torch.LongTensor(ends).contiguous()}


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
            data = {k:v.to(self.device) for k, v in data.items()}
            loss, start_token, end_token = model(data)
            
            f11 = compute_f1(start_token, data["starts"])
            f12 = comput_f1(end_token, data["ends"])
            f1 = (f11+f12)/2

            if self.type =="train":
                self.optim_process(model, loss)
                self.step += 1
                self.writer.add_scalar("train/f1", f1.data, self.step)

            else:
                total_loss.append(loss.item())
                total_f1.append(f1.item())
                
        if self.type != "train":
            self.writer.add_scalar("test/loss", sum(total_loss)/len(total_loss), self.step)
            self.writer.add_scalar("test/ppl", torch.exp(sum(total_accuracy)/len(total_loss)), self.step)
            self.writer.add_scalar("test/f1", sum(total_f1)/len(total_f1), self.step)
            print("ppl : {}".format(torch.exp(sum(total_accuracy)/len(total_loss))))
            print("f1 : {}".format(sum(total_f1)/len(total_f1)))

    def optim_process(self, model, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.log_writer(loss.data, self.step)


            










        
        



