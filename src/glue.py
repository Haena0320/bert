import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys, os
sys.path.append(os.getcwd())

from src.metrics import compute_f1
from tqdm import tqdm

class Classification_Task(nn.Module):
    def __init__(self, config, device, model, num_labels, type):
        self.config = config
        self.device = device
        self.model = model
        self.type = type
        
        self.dropout = nn.Dropout(config.model.d_rate)
        self.classifier = nn.Linear(config.model.hidden, num_labels)
        if self.type == "regression":
            self.loss_fn = nn.MSELoss()

        if self.type =="single_class":
            self.loss_fn = nn.CrossEntropyLoss()

        if self.type =="multi_class":
            self.loss_fn = BCEWithLogitsLoss() #???


    def forward(self, data):
        hidden = self.model(data['inputs'], data["segments"])
        C = hidden[0] # hidden node vector of cls token
        logits = self.classifier(self.dropout(C))

        if self.type == "regression":
            loss = self.loss_fn(logits.view(-1, self.num_labels), data["labels"])

        if self.type == "single_label_classification":
            loss = self.loss_fn(logits.view(-1, self.num_labels), data["labels"])

        if self.type == "multi_label_classification":
            loss = self.loss_fn(logits, data["label"])
        return loss

class Question_Answering_Task(nn.Module):
    def __init__(self, config, device, model):
        self.config = config
        self.device = device
        self.model = model

        self.activation = F.gleu()
        self.qa_span = nn.Linear(config.model.hidden, 2)

    def forward(self, data):
        output = self.model(data["input"], data["segment"])
        output = self.activation(output)
        pos_output = self.qa_span(output)
        start_pos, end_pos = pos_output.split(1, dim=-1)
        start_pos = start_pos.squeeze(-1)
        end_pos = end_pos.squeeze(-1)
        return start_pos, end_pos


def get_optimizer(model, args_optim, lr):
    if args_optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-09)


def get_trainer(config, args, device, data_loader, sp, writer, type):
    return Trainer(config, args, device, data_loader, sp, writer, type)


class Trainer:
    def __init__(self, config, args, device, data_loader, sp, writer, type):
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
        if self.type == "train":
            self.writer.add_scalar('train/loss', log, self.global_step)
            self.writer.add_scalar('train/lr', lr, self.global_step)

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
        total_f1 = list()

        for data in tqdm(self.data_loader):
            data = {k: v.to(self.device) for k, v in data.items()}
            squad_output = model(data)
            loss, accuracy = get_loss(squad_output, data["label"])

            f1 = compute_f1(squad_output, data["label"], data['answers'])

            if self.type == "train":

                self.optim_process(model, loss)
                self.step += 1
                self.writer.add_scalar("train/accuracy", accuracy.data, self.step)
                self.writer.add_scalar("train/f1", f1.data, self.step)

                if self.step % self.ckpnt_step == 0:
                    torch.save({"epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict()},
                               save_path + "cknpt_{}".format(epoch))
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

