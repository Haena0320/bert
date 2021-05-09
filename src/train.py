import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import tqdm as tqdm

def get_trainer(config, args, device, data_loader, writer, type):
    return Trainer(config, args, device, data_loader, writer, type)

def get_optimizer(model, args_optim):
    if args_optim =="adam":
        return torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-09)

def get_lr_schedular(optimizer, config):
    hidden = config.model.hidden
    warmup =config.train.warmup
    return WarmupLinearschedular(optimizer, hidden, warmup)

class WarmupLinearschedular:
    def __init__(self, optimizer, hidden, warmup):
        self.optimizer = optimizer
        self._step=0
        self.warmup=warmup
        self.hidden = hidden
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] =  rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.hidden **(-.5)*min(step**(-.5), step*self.warmup**(-1.5))


class Trainer:
    def __init__(self, config, args, device, data_loader, writer, type):
        self.config = config
        self.args = args
        self.device = device
        self.data_loader = data_loader
        self.writer = writer
        self.type=type
        self.accum = config.train.accum_stack
        self.ckpnt = config.train.ckpnt_step
        self.global_step = 0
        self.train_loss = 0

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer

    def init_scheduler(self, scheduler):
        self.scheduler = scheduler

    def log_writer(self, log, step):
        if self.type =="train":
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/loss", log, self.global_step)
            self.writer.add_scalar("train/lr", lr, self.global_step)
        else:
            self.writer.add_scalar("valid/loss", log, step)

    def train_epoch(self, model, epoch, global_step=None, save_path=None):
        if self.type=="train":
            model.train()

        else:
            model.eval()

        model.to(self.device)
        loss_save = list()

        data_iter = tqdm.tqdm(enumerate(train_loader), desc="%s_data: %d" % (self.type, epoch), total=len(self.data_loader), bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            data
            break



