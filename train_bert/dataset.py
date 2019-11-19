# -*- coding: utf-8 -*-
import logging
import json
import torch
from tqdm import tqdm
import torch.utils.data as data
from data_util import config

logger = logging.getLogger("restoration")

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
                    "target": [0] + sample["tagging"] + [0] * (config.max_len-len(sample["tagging"])-1)
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

        # Modify tokens to a fixed length
        doc_len = len(tokens)
        if doc_len < config.max_len:
            tokens.extend(["[PAD]"] * (config.max_len - doc_len))
        else:
            tokens = tokens[:config.max_len]
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokens)

        return indexed_tokens

    def data_to_bert_input(self, cells):
        features = []
        targets = []
        with tqdm(total=len(cells)) as pbar:
            for cell in cells:
                features.append(self.feature2vec(cell["feature"]))
                targets.append(cell["target"])
                pbar.update(1)
                
        # Show some examples to make sure data is handled in a right way
        ori = self.bert_tokenizer.convert_ids_to_tokens(features[0])
        logger.info("Length of feature: {}".format(len(features[0])))
        logger.info("Length of target : {}".format(len(targets[0])))
        for idx, tag in enumerate(targets[0]):
            if tag == 1:
                logger.info("{}".format(ori[idx]))

        self.features = features
        self.targets = targets

    def get_dataset(self):
        return Dataset(self.features, self.targets)

