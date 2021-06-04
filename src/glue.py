import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys, os
sys.path.append(os.getcwd())

from src.metrics import *
from tqdm import tqdm

class Classification_Task(nn.Module):
    def __init__(self, config, device, model, num_labels, type):
        super(Classification_Task, self).__init__()
        self.config = config
        self.device = device
        self.model = model
        self.type = type
        self.num_labels = num_labels
        
        self.dropout = nn.Dropout(config.model.d_rate)
        self.classifier = nn.Linear(config.model.hidden, num_labels)
        if self.type == "regression":
            self.loss_fn = nn.MSELoss()

        if self.type =="single_class":
            self.loss_fn = nn.CrossEntropyLoss()

        if self.type =="multi_class":
            self.loss_fn = nn.CrossEntropyLoss() #???


    def forward(self, data):
        hidden = self.model(data['inputs'], data["segments"])
        C = hidden[:,0, :] # hidden node vector of cls token
        logits = self.classifier(self.dropout(C))
        loss = 0
        if self.type == "regression":
            loss = self.loss_fn(logits.view(-1, self.num_labels), data["labels"])

        elif self.type == "single_class":
            loss = self.loss_fn(logits.view(-1, self.num_labels), data["labels"])

        elif self.type == "multi_class":
            loss = self.loss_fn(logits, data["labels"])
        else:
            print(self.type)
            print("this type is not supported ! ")
        return loss, logits

class Question_Answering_Task(nn.Module):
    def __init__(self, config, device, model):
        super(Question_Answering_Task, self).__init__()
        self.config = config
        self.device = device
        self.model = model

        self.activation = nn.GELU()
        self.qa_span = nn.Linear(config.model.hidden, 2)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, data):
        output = self.model(data["inputs"], data["segments"])
        output = self.activation(output)
        pos_output = self.qa_span(output)
        start_pos, end_pos = pos_output.split(1, dim=-1)
        start_pos = start_pos.squeeze(-1)
        end_pos = end_pos.squeeze(-1)

        start_label =data["starts"]
        end_label = data["ends"]
        print(start_pos)
        print(start_label)
        print(end_pos)
        print(end_label)
        bs, seq = start_pos.size()
        print(start_pos.view(-1).size())
        print(start_pos.size())
        print(start_label.size())
        loss = self.criterion(start_pos.view(-1), start_label.view(-1))+self.criterion(end_pos.view(-1), end_label.view(-1))
        return loss, start_pos, end_pos

def get_optimizer(model, args_optim, lr):
    if args_optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-09)

def get_trainer(config, args, device, data_loader, sp, writer, type, metric):
    return Trainer(config, args, device, data_loader, sp, writer, type, metric)


class Trainer:
    def __init__(self, config, args, device, data_loader, sp, writer, type, metric):
        self.config = config
        self.args = args
        self.device = device
        self.writer = writer
        self.type = type
        self.data_loader = data_loader
        self.sp = sp
        self.metric = metric

        self.step = 0
        self.clip = 5

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer

    def init_scheduler(self, scheduler):
        self.scheduler = scheduler

    def log_writer(self, log, step):
        if self.type == "train":
            self.writer.add_scalar('train/loss', log, self.step)

        else:
            self.writer.add_scalar("valid/loss", log, step)

    def train_epoch(self, model, epoch, save_path=None):
        if self.type == "train":
            model.train()

        else:
            model.eval()
        model.to(self.device)
        total_loss = list()
        total_accuracy = list()
        total_score = list()
        f = open(save_path, "w")
        cnt =0
        for data in tqdm(self.data_loader):
            data = {k: v.to(self.device) for k, v in data.items()}
            cnt +=1
            loss, glue_output = model(data)

            score = self.compute_metric(self.metric, glue_output, data["labels"])
            accuracy = self.compute_metric("Accuracy",glue_output, data["labels"])

            if self.type == "train":
                self.optim_process(model, loss)
                self.step += 1
                self.writer.add_scalar("train/metric", score, self.step)
                self.writer.add_scalar("train/accuracy", accuracy.data, self.step)
                f.write('train mode | batch {:3d} | loss {:8.5f} | score {} | accuracy '.format(cnt + 1, loss, score,accuracy.item()) + '\n')

            else:
                total_loss.append(loss.item())
                total_score.append(score.item())
                total_accuracy.append(accuracy.item())
                f.write('test mode | batch {:3d} | loss {:8.5f} | score {} | accuracy '.format(cnt + 1, loss, score, accuracy.item()) + '\n')

        if self.type != "train":
            self.writer.add_scalar("test/loss", sum(total_loss)/len(total_loss), self.step)
            self.writer.add_scalar("test/score", sum(total_score)/len(total_score), self.step)
            self.writer.add_scalar("test/accuracy", sum(total_accuracy)/len(total_accuracy), self.step)
            print("test score : {}".format(sum(total_score)/len(total_score)))
            print("test accuracy : {}".format(sum(total_accuracy)/len(total_accuracy)))
            return sum(total_score)/len(total_score), sum(total_accuracy)/len(total_accuracy)

    def optim_process(self, model, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.log_writer(loss.data, self.step)

    def compute_metric(self, m, pred, label):
        if m == "Accuracy":
            return compute_accuracy(pred, label)
        elif m =="Pearson_cor":
            return compute_Pearson_corr(pred, label)
        elif m == "F1":
            return compute_f1(pred, label)
        elif m =="Mat_corr":
            return compute_Mat_corr(pred, label)
        else:
            print("this metric is not supported ! ")




