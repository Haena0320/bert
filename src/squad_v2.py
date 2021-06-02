import sys, os
import json
from pathlib import Path


def squad_v2_prepro(raw_path, save_path, sp, seq_len=128):
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
                if len(answer_starts) == 0:
                    print(qa['question'])
                    print(context)
                    no_answer_cnt += 1
                    continue
                else:
                    answer_starts = answer_starts[0]
                answers = [sp.EncodeAsIds(answer["text"].strip()) for answer in qa["answers"]]

                _input = [bos] + question + [eos] + context
                _segment = [1] + [1] * len(question) + [1] + [2] * len(context)
                _input += [pad] * (seq_len - len(_input))
                _segment += [pad] * (seq_len - len(_segment))

                _label = [answer_starts] + [answer_starts + len(answers) - 1]  # start, end token

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
    print(context)
    print(item)
    print("no anwser question : {}".format(no_answer_cnt))
    print('finished !! ')

    return None
    
