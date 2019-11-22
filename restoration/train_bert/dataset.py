# -*- coding: utf-8 -*-
import logging
import json
import torch
from tqdm import tqdm
import torch.utils.data as data
from restoration.data_util import config

logger = logging.getLogger("restoration")


class Dataset(data.Dataset):

    def __init__(self, features, targets, segments, attns):
        self.targets = torch.tensor(targets)
        self.features = torch.tensor(features)
        self.segments = torch.tensor(segments, dtype=torch.long)
        self.attns = torch.tensor(attns, dtype=torch.long)
        assert len(targets) == len(features) == len(segments) == len(attns)
        self.size = len(targets)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.features[index], self.targets[index], self.segments[index], self.attns[index]


class Data(object):

    def __init__(self, bert_tokenizer):
        self.bert_tokenizer = bert_tokenizer
        self.features = None
        self.targets = None
        self.segments = None
        self.attns = None

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
        segment_tokens = [0]
        sents = sent.split(" ")
        tokens.append("[CLS]")

        # print(len(sents), sents)
        assert len(sents) == 5

        for idx, a_sent in enumerate(sents):
            a_token = self.bert_tokenizer.tokenize(a_sent)
            tokens.extend(a_token)
            tokens.append("[SEP]")
            if idx+1 != 5:
                segment_tokens.extend([0] * (len(a_token) + 1))
            else:
                segment_tokens.extend([1] * (len(a_token) + 1))

        # Attention tokens for ignoring padding
        attn_tokens = [1] * len(tokens)

        # Modify tokens to a fixed length
        doc_len = len(tokens)
        if doc_len < config.max_len:
            tokens.extend(["[PAD]"] * (config.max_len - doc_len))
            segment_tokens.extend([1] * (config.max_len - doc_len))
            attn_tokens.extend([0] * (config.max_len - doc_len))
        else:
            tokens = tokens[:config.max_len]
            segment_tokens = segment_tokens[:config.max_len]
            attn_tokens = attn_tokens[:config.max_len]
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokens)

        return indexed_tokens, segment_tokens, attn_tokens

    def tag_to_word(self, indexed_tokens, tags):
        if isinstance(indexed_tokens, torch.Tensor):
            indexed_tokens = indexed_tokens.to("cpu")
            indexed_tokens = indexed_tokens.tolist()
        ori = self.bert_tokenizer.convert_ids_to_tokens(indexed_tokens)
        logger.info("Origin Sentence  : {}".format(" ".join(ori)))
        words = []
        for idx, tag in enumerate(tags):
            if tag == 1:
                logger.info("[Position {}]{}".format(idx, ori[idx]))
                words.append((idx, ori[idx]))
        return words

    def one_data_to_bert_input(self, cell):
        indexed_tokens, segment_tokens, attn_tokens = self.feature2vec(cell["feature"])

        return torch.tensor(indexed_tokens), \
               torch.tensor(segment_tokens, dtype=torch.long), \
               torch.tensor(attn_tokens, dtype=torch.long)

    def data_to_bert_input(self, cells):
        features = []
        segments = []
        attns = []
        targets = []
        with tqdm(total=len(cells)) as pbar:
            for cell in cells:
                indexed_tokens, segment_tokens, attn_tokens = self.feature2vec(cell["feature"])
                features.append(indexed_tokens)
                segments.append(segment_tokens)
                attns.append(attn_tokens)
                targets.append(cell["target"][:config.max_len])
                pbar.update(1)

        # Show the first sample to make sure data is handled in a right way
        logger.info("Length of feature: {}".format(len(features[0])))
        logger.info("Length of target : {}".format(len(targets[0])))
        self.tag_to_word(features[0], targets[0])

        self.features = features
        self.targets = targets
        self.segments = segments
        self.attns = attns

    def get_dataset(self):
        return Dataset(self.features, self.targets, self.segments, self.attns)

