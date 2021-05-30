import sys, os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda import amp
from src.prepro import *

from tqdm import tqdm

def get_trainer(config, args, device, file_list, sp, writer, type):
    return Trainer(config, args, device, file_list,sp, writer, type)

def get_optimizer(model, args_optim):
    if args_optim =="adam":
        return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-09)

def get_lr_scheduler(optimizer, config):
    hidden = config.model.hidden
    warmup =config.pretrain.warmup
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
            p["lr"] = rate
        self._rate = rate

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.hidden **(-.5)*min(step**(-.5), step*self.warmup**(-1.5))


class Trainer:
    def __init__(self, config, args, device, file_list, sp, writer, type):
        self.config = config
        self.args = args
        self.device = device
        self.file_list = file_list
        self.writer = writer
        self.type=type
        self.sp = sp
        self.accum = config.pretrain.accum_stack
        self.ckpnt_step = config.pretrain.ckpnt_step
        self.gradscaler = amp.GradScaler()
        self.global_step = 0
        self.step = 0
        self.train_loss = 0

        self.get_loss = nn.CrossEntropyLoss(ignore_index=0)

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
        total_correct = 0
        correct_t = 0

        for f in tqdm(self.file_list):
            data_loader = BERTDataloader(self.config, f, self.sp)
            for data in tqdm(data_loader):
                data = {k:v.to(self.device) for k, v in data.items()}

                mask_lm_output, next_sent_output= model.forward(data["bert_input"], data["segment_input"])
                bs, seq, _ = mask_lm_output.size()

                accuracy = torch.argmax(next_sent_output, dim=-1)
                accuracy= sum(accuracy.eq(data["is_next"]))*100/len(data["is_next"])
                self.writer.add_scalar("next/accuracy",accuracy.data, self.step)

                next_loss = self.get_loss(next_sent_output, data["is_next"])
                mask_loss = self.get_loss(mask_lm_output.view(bs*seq, -1), data["bert_label"].view(-1))
                loss = next_loss + mask_loss

                if self.type =="train":
                    self.optim_process(model, loss)
                    self.step += 1

                    if self.step % self.ckpnt_step ==0:
                        torch.save({"epoch":epoch,
                                    "model_state_dict":model.state_dict(),
                                    "optimizer_state_dict":self.optimizer.state_dict(),
                                    "lr_step":self.scheduler._step},
                                   save_path+"ckpnt_{}".format(epoch))

                else:
                    loss_save.append(loss.item())

                    #next sentence accuracy
                    correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).long()
                    correct_t += len(data["is_next"])
                    total_correct += correct.sum().item()

            if self.type != "train":
                te_loss = sum(loss_save)/len(loss_save)
                self.writer.add_scalar("test/loss",te_loss, self.step)
                self.writer.add_scalar("test/accuracy", total_correct*100/correct_t, self.step)

            else:
                pass

            del data_loader
        self.writer.close()

    def optim_process(self, model, loss):
        loss /= self.accum
        #self.gradscaler.scale(loss).backward()
        loss.backward()

        if self.step % self.accum == 0:
            #self.gradscaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.pretrain.clip)
            #self.gradscaler.step(self.optimizer)
            #self.gradscaler.update()
            self.scheduler.step()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.log_writer(loss.data*self.accum, self.global_step)
            self.global_step += 1



