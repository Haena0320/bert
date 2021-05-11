import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp
import json
import glob

def setup(rank, world_size, seed):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("ncc1", rank=rank, word_size=world_size)

    torch.manual_seed(seed)

def cleanup():
    dist.destroy_process_group()

def run_demo(demo_fn, args):
    mp.spawn(demo_fn, args=(args,), nprocs = args.world_size, join=True)

def print_loss_log(file_name, train_loss, val_loss, test_loss, args=None):
    with open(file_name, "w") as f:
        if args:
            for item in args.__dict__:
                f.write(item+":   "+str(args.__dict__[item])+"\n")
        for idx in range(len(train_loss)):
            f.write('epoch {:3d} | train loss {:8.5f}'.format(idx+1, train_loss[idx])+'\n')

        for idx in range(len(val_loss)):
            f.write("epoch {:3d} | val loss {:8.5f}".format(idx+1, val_loss[idx])+"\n")

        f.write('test loss {:8.5f}'.format(test_loss)+'\n')


class Dictobj(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def __getitem__(self, key):
        return getattr(self, key)

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Dictobj(value) if isinstance(value, dict) else value

def load_config(conf):
    with open(os.path.join("config", "{}.json".format(conf)), "r") as f:
        config = json.load(f)
    return Dictobj(config)
