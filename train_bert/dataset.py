# -*- coding: utf-8 -*-
import logging
import json
import torch
from tqdm import tqdm
import torch.utils.data as data
from data_util import config


class Dataset(data.Dataset):

    def __init__(self, features, targets):
        self.targets = torch.tensor(targets)
        self.features = torch.tensor(features)
        assert len(targets) == len(features)
        self.size = len(targets)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


class Data(object):

    def __init__(self, bert_tokenizer):
        self.bert_tokenizer = bert_tokenizer
        self.features = None
        self.targets = None

    def load_from_file(self, filename):
        # The structure of data looks like:
        # {
        # "context": "... ... ... ...",
        # "utterance": "...",
        # "tagging": [0,0,0,..., 1,1, ...., 0],
        # "rewritten": "..."
        # }

        samples = []

        with open(filename, 'r', encoding='utf8') as file:
            json_data = file.read()
            data = json.loads(json_data)
            for sample in data:
                a_train_sample = {
                    "feature": sample["context"] + " " + sample["utterance"],
                    "target": sample["tagging"] + [0] * (config.max_len-len(sample["tagging"]))
                }
                samples.append(a_train_sample)
        return samples

    def feature2vec(self, sent):
        # Space character will be turned into <SEP>
        tokens = []
        sents = sent.split(" ")
        tokens.append("[CLS]")
        for a_sent in sents:
            tokens.extend(self.bert_tokenizer.tokenize(a_sent))
            tokens.append("[SEP]")

        doc_len = len(tokens)

        if doc_len < config.max_len:
            tokens.extend(["[PAD]"] * (config.max_len - doc_len))
        else:
            tokens = tokens[:config.max_len]
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        # print(len(indexed_tokens), indexed_tokens)
        return indexed_tokens

    def data_to_bert_input(self, cells):
        features = []
        targets = []
        with tqdm(total=len(cells)) as pbar:
            for cell in cells:
                features.append(self.feature2vec(cell["feature"]))
                targets.append(cell["target"])
                pbar.update(1)
        self.features = features
        self.targets = targets

    def get_dataset(self):
        return Dataset(self.features, self.targets)

