import collections
import re
import string
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef
import numpy as np
import torch

def compute_qa_exact(ans_pred_tokens_samples):
    """
    :param ans_pred_tokens_samples: [([ans1_token_candidate1, ans1_token_candidate2], pred1_tokens),
                                    ([ans2_token_candidate1, ans2_tokens_candidate2], pred2_tokens),
                                    ([ansn_tokens_candidate1, ansn_tokens_candidate2], predn_tokens)]
                                     
    :return: exact score of the samples
    """
    
    def normalize_txt(text):
        text = text.lower()
        
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)
        regex = re.complie(r"\b(a|an|the)\b", re.UNICODE)
        text = re.sub(regex, " ", text)

        return " ".join(text.split())

    exact_scores = []
    for (ans_tokens, pred_tokens) in ans_pred_tokens_samples:
        pred_str = " ".join(pred_tokens)
        candidate_score = []
        for item in ans_tokens:
            ans_str = " ".join(item)
            candidate_score.append(int(normalize_txt(ans_str)==normalize_txt(pred_str)))
        exact_scores.append(max(candidate_score))
    return 100.0*sum(exact_scores)/len(exact_scores)

def compute_qa_f1(ans_pred_tokens_samples):
    """
    :param ans_pred_tokens_samples: [([ans1_token_candidate1, ans1_token_candidate2], pred1_tokens),
                                    ([ans2_token_candidate1, ans2_tokens_candidate2], pred2_tokens),
                                    ([ansn_tokens_candidate1, ansn_tokens_candidate2], predn_tokens)]

    :return: f1 score of the samples
    """
    def sample_f1(ans_tokens, pred_tokens):
        common = collections.Counter(ans_tokens) & collections.Counter(pred_tokens)
        num_same = sum(common.values())
        if len(ans_tokens) == 0 or len(pred_tokens) == 0:
            return int(ans_tokens == pred_tokens)

        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(ans_tokens)
        f1 = (2*precision*recall)/(precision+recall)
        return f1

    f1_scores = []
    for idx in range(len(answers)):
        for (ans_tokens, pred_tokens) in ans_pred_tokens_samples:
            candidate_score = []
            for item in ans_tokens:
                candidate_score.append(sample_f1(item, pred_tokens))
            f1_scores.append(max(candidate_score))

    return 100.0*sum(f1_scores)/len(f1_scores)

def compute_squad_f1(pred, label ,answers):
    """
    :param pred: bs*2 (start_pred, end_pred)
    :param label: bs*2 (start_label, end_label)
    :param answers: bs, [paragrapth_id, 1,2,3,6,4]
    :return:
    """
    def sample_f1(pred_1, label_1, answer_1):
        sp =pred_1[0]
        ep = pred_1[1]
        sl = label_1[0]
        el = label_1[1]

        pred_token = answer_1[sp:ep+1]
        label_token = answer_1[sl:el+1]
        common = collections.Counter(pred_token) & collections.Counter(label_token)
        num_same = sum(common.values())
        if num_same ==0:
            return 0
        precision = num_same / len(pred_token)
        recall = num_same / len(label_token)
        f1 = 2*(precision*recall) /(precision+recall)
        return f1

    f1_score = []
    for idx in range(len(label)):
        pred_sample = pred[idx]
        label_sample = label[idx]
        answer_sample = answers[idx]
        f = sample_f1(pred_sample,label_sample, answer_sample)
        f1_score.append(f)

    return 100*sum(f1_score)/len(f1_score)

def compute_accuracy(pred, label):
    pred = torch.argmax(pred, dim=-1)
    return sum((pred==label).float()) / len(label)
    # return sum(pred.eq(label).float())/len(label)
def compute_f1(pred, label):
    pred = torch.argmax(pred, dim=-1)
    tp = (pred*label).sum()
    tn = ((1-label)*(1-pred)).sum()
    fp = ((1-label)*pred).sum()
    fn = (label*(1-pred)).sum()
    precision = tp / (tp+fp+1e-7)
    recall = tp / (tp+fn+1e-7)
    f1 = 2*precision*recall/(precision+recall+1e-7)
    return f1

def compute_Mat_corr(pred, label):
    pred = torch.argmax(pred, dim=-1)
    tp = (pred * label).sum()
    tn = ((1 - label) * (1 - pred)).sum()
    fp = ((1 - label) * pred).sum()
    fn = (label * (1 - pred)).sum()
    mcc = (tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-7)**2
    return mcc

def compute_Pearson_corr(pred, label):
   # pred : (bs, 1)
   # label : (bs, 1)
   pred = torch.argmax(pred, dim=-1)
   corr, p_value = pearsonr(pred, label)
   return corr





        
    