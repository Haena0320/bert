import torch
import csv
# CoLA ['gj04', '1', '', 'The sailors rode the breeze clear of the rocks.']
def prepro_1(dataset="CoLA", file_path=None, save_path=None, type='train',encoding="utf-8", sp=None, seq_len=128, line=1):
    data = dict()
    data["inputs"] = list()
    data["segments"] = list()
    pad = 0
    bos = 1
    eos = 2

    f = open(file_path, "r", encoding=encoding)
    rdr = csv.reader(f, delimiter="\t")
    r = list(rdr)
    print("{}--{}---------------------------------------------------------------".format(dataset, type))
    if type != "test":
        data['labels'] = list()

    for i in range(len(r)):
        if type != "test":
            label = int(r[i][1])
            data['labels'].append(label)

        if dataset =="CoLA":
            input =r[i][-1]
            input =[bos] + sp.EncodeAsIds(input)+[eos]
            input_ = input+[pad]*(seq_len-len(input))

        if dataset =='SST_2':
            input = r[i][0]
            input = [bos]+sp.EncodeAsIds(input)+[eos]
            input_ = input + [pad]*(seq_len-len(input))

        assert len(input_) == seq_len
        data["inputs"].append(input_)
        segment = [1]*seq_len
        data["segments"].append(segment)

    assert len(data["inputs"]) ==len(data["segments"])

    print("{} prepro data size : {}".format(dataset, len(data["inputs"])))
    print("{} prepro sample :".format(dataset))
    print("input : {}".format(input_))
    print("segment : {}".format(segment))

    if type != "test":
        print("label : {}".format(label))
        assert len(data["inputs"]) ==len(data["labels"])

    torch.save(data, save_path)
    return None



def prepro_2(dataset="MRPC", file_path=None, save_path=None, type='train',encoding="utf-8", sp=None, seq_len=128, line=1):
    data = dict()
    data["inputs"] = list()
    data["segments"] = list()
    pad = 0
    bos = 1
    eos = 2

    f = open(file_path, "r", encoding=encoding)
    rdr = csv.reader(f, delimiter="\t")
    r = list(rdr)
    print("{}--{}---------------------------------------------------------------------------------------------------------------------------".format(dataset, type))

    if type != "test":
        data['labels'] = list()

    for i in range(len(r)):
        if type != "test":
            if dataset =="MRPC":
                label = int(r[i][0])
                input_1 = r[i][-2]
                input_2 = r[i][-1]
                
            elif dataset in ["QQP", "STS_B","WNLI"]:
                label = int(r[i][-1])
                input_1 = r[i][-3]
                input_2 = r[i][-2]
                
            elif dataset =="QNLI":
                dict_ = {"entailment":0, "not_entailment":1}
                label = r[i][-1]
                assert label in dict_
                label = dict_[label]
                input_1 = r[i][1]
                input_2 = r[i][2]

            elif dataset =="RTE":
                label = r[i][-1]
                dict_ = {"entailment":0, "not_entailment":1}
                label = dict_[label]
                input_1 = r[i][-3]
                input_2 = r[i][-2]

            elif dataset in ["MNLI_m","MNLI_mm"]:
                dict_ = {"entailment":0, "neutral":1, "contradiction":2}
                label = r[i][-1]
                label = dict_[label]
                input_1 = r[i][8]
                input_2 = r[i][9]
                
            data['labels'].append(label)

        else:
            input_1 = r[i][-2]
            input_2 = r[i][-1]

        input_1 =[bos] + sp.EncodeAsIds(input_1)+[eos]
        input_2 = sp.EncodeAsIds(input_2)
        input_ = input_1+input_2
        segment = len(input_1) * [1] + len(input_2) * [2]
        if len(input_) > seq_len:
            input_ = input_[:seq_len]
            segment = segment[:seq_len]
        else:
            input_ = input_ + [pad]*(seq_len-len(input_))
            segment = segment + [pad]*(seq_len-len(segment))

        assert len(input_) == seq_len
        data["inputs"].append(input_)
        data["segments"].append(segment)

    assert len(data["inputs"]) ==len(data["segments"])

    print("{} prepro data size : {}".format(dataset, len(data["inputs"])))
    print("{} prepro sample :".format(dataset))
    print("input : {}".format(input_))
    print("segment : {}".format(segment))
    if type != "test":
        print("label : {}".format(label))

        assert len(data["inputs"]) == len(data['labels'])
    torch.save(data, save_path)
    return None

#######################################

class GLUE_Dataset(Dataset):
    def __init__(self, filepath):
        logging.info("generating examples from = %s", filepath)
        self.data = torch.load(filepath)["data"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        inputs = self.data[item]
        return {k:torch.LongTensor(v) for k,v in inputs.items()}

def GLUE_Loader(corpus_path, bs=32, num_workers=3, shuffle=True,drop_last=True):
    dataset = GLUE_Dataset(corpus_path)
    data_loader = DataLoader(dataset, batch_size=bs, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=padd_fn)
    return data_loader