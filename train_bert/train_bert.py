import logging
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from model import FeedForwardNeuralNetwork
from dataset import Data
from data_util import config

logger = logging.getLogger("restoration")


class Trainee(object):
    def __init__(self, bert_model):
        super().__init__()
        self.device, n_gpu = self._check_device()
        self.bert_model = bert_model.to(self.device)
        self.ffnn_model = FeedForwardNeuralNetwork({
            "class-number": 2,
            "hidden-dimension": 768,
            "dropout-rate": 0.5
        }).to(self.device)

    def _check_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "cuda":
            n_gpu = torch.cuda.device_count()
            return device, n_gpu
        return device, -1

    def train(self, data, fine_tune=False, save_dir=None):
        # Initialize path
        loss_stats = None
        if save_dir:
            if not os.path.exists(save_dir):
                raise FileNotFoundError("save_dir is specified but not found: {}".format(save_dir))
            loss_stats = os.path.join(save_dir, "loss.txt")

        # Initialize data loader
        data_loader = DataLoader(data.get_dataset(),
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 drop_last=True)

        # Initialize optimizers of BERT and FFNN
        bert_parameters = [p for n, p in list(self.bert_model.named_parameters())]
        if fine_tune:
            classifier_parameters = list(bert_parameters + self.ffnn_model.parameters())
        else:
            classifier_parameters = list(self.ffnn_model.parameters())
        optimizer = torch.optim.Adam(classifier_parameters, lr=config.lr)

        # Per epoch 
        for epoch in range(config.total_epochs):
            num_of_batch = len(data_loader.dataset) // config.batch_size
            cnt = 1
            # Per batch
            with tqdm(total=num_of_batch) as pbar:
                for feature, target in data_loader:
                    # Checkout device
                    feature, target = feature.to(self.device), target.to(self.device)

                    optimizer.zero_grad()

                    # BERT part
                    segments_ids = np.zeros(list(feature.shape))
                    segments_tensors = torch.tensor(segments_ids, dtype=torch.long)
                    segments_tensors = segments_tensors.to(self.device)
                    if fine_tune:
                        self.bert_model.train()
                    else:
                        self.bert_model.eval()
                    encoded_layers, _ = self.bert_model(feature, segments_tensors)
                    encoded_layers.to(self.device)

                    # Tag classifier
                    self.ffnn_model.train()
                    tag_loss = None
                    for idx in range(0, config.max_len):
                        indices = torch.tensor([idx], dtype=torch.long).to(self.device)
                        cls_feature = torch.index_select(encoded_layers, 1, indices)
                        tag = self.ffnn_model(cls_feature)

                        if tag_loss is None:
                            tag_loss = F.cross_entropy(tag, target[:, idx])
                        else:
                            tag_loss += F.cross_entropy(tag, target[:, idx])

                    # Loss output
                    tag_loss = tag_loss / config.max_len
                    tqdm.write("[{}/{}] LOSS = {}".format(cnt, num_of_batch, tag_loss.item()))

                    if loss_stats and save_dir:
                        if not os.path.exists(loss_stats):
                            with open(loss_stats, 'w', encoding='utf8') as file:
                                file.write("{}\t{}\t{}".format(num_of_batch, cnt, tag_loss.item()))
                                file.write("\n")
                        else:
                            with open(loss_stats, 'a', encoding='utf8') as file:
                                file.write("{}\t{}\t{}".format(num_of_batch, cnt, tag_loss.item()))
                                file.write("\n")

                    tag_loss.backward()

                    # Step
                    optimizer.step()
                    cnt += 1
                    pbar.update(1)

    def test(self, data, save_dir=None):
        # Initialize path
        loss_stats = None
        if save_dir:
            if not os.path.exists(save_dir):
                raise FileNotFoundError("save_dir is specified but not found: {}".format(save_dir))
            loss_stats = os.path.join(save_dir, "loss.txt")

        # Initialize data loader
        data_loader = DataLoader(data.get_dataset(),
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 drop_last=True)
        num_of_batch = len(data_loader.dataset) / config.batch_size
        cnt = 1
        with tqdm(total=num_of_batch) as pbar:
            for feature, target in data_loader:
                # Checkout device
                feature, target = feature.to(self.device), target.to(self.device)

                # BERT part
                segments_ids = np.zeros(list(feature.shape))
                segments_tensors = torch.tensor(segments_ids, dtype=torch.long)
                segments_tensors = segments_tensors.to(self.device)
                self.bert_model.eval()
                encoded_layers, _ = self.bert_model(feature, segments_tensors)
                encoded_layers = encoded_layers.to(self.device)

                # Tag classifier
                self.ffnn_model.eval()
                tag_loss = None
                for idx in range(0, config.max_len):
                    indices = torch.tensor([idx], dtype=torch.long)
                    indices = indices.to(self.device)
                    cls_feature = torch.index_select(encoded_layers, 1, indices)
                    tag = self.ffnn_model(cls_feature)

                    if tag_loss is None:
                        tag_loss = F.cross_entropy(tag, target[:, idx])
                    else:
                        tag_loss += F.cross_entropy(tag, target[:, idx])

                # Loss output
                tag_loss = tag_loss / config.max_len
                tqdm.write("[{}/{}] LOSS = {}".format(cnt, num_of_batch, tag_loss.item()))

                if loss_stats and save_dir:
                    if not os.path.exists(loss_stats):
                        with open(loss_stats, 'w', encoding='utf8') as file:
                            file.write("{}\t{}\t{}".format(num_of_batch, cnt, tag_loss.item()))
                            file.write("\n")
                    else:
                        with open(loss_stats, 'a', encoding='utf8') as file:
                            file.write("{}\t{}\t{}".format(num_of_batch, cnt, tag_loss.item()))
                            file.write("\n")

                # Step
                cnt += 1
                pbar.update(1)

    def load_model(self, filename):
        if torch.cuda.is_available():
            model_state = torch.load(filename)
        else:
            model_state = torch.load(filename, map_location="cpu")

        self.ffnn_model.load_state_dict(model_state, strict=False)

    def save_model(self, filename):
        torch.save(self.ffnn_model.state_dict(), filename)

    def save_bert(self, dir_name):
        if not self.bert_model:
            raise ValueError('Intend to save bert model while bert model is None')
        if not os.path.exists(dir_name):
            raise FileNotFoundError("Specified Path is not found: {}".format(dir_name))

        bert_file = os.path.join(dir_name, "pytorch_model.bin")
        bert_config = os.path.join(dir_name, "bert_config.json")
        model_to_save = self.bert_model.module if hasattr(self.bert_model, 'module') else self.bert_model

        to_save_dict = model_to_save.state_dict()
        to_save_with_prefix = {}
        for key, value in to_save_dict.items():
            to_save_with_prefix['bert.' + key] = value
        torch.save(to_save_with_prefix, bert_file)
        logger.info("save bert model to {}".format(bert_file))
        with open(bert_config, 'w') as f:
            f.write(model_to_save.config.to_json_string())


if __name__ == "__main__":
    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_file)
    bert_model = BertModel.from_pretrained(config.bert_file)

    # Data
    data = Data(bert_tokenizer=bert_tokenizer)
    samples = data.load_from_file(config.data_file)
    data.data_to_bert_input(samples)

    # Training
    trainee = Trainee(bert_model=bert_model)
    trainee.train(data=data, fine_tune=False)

    # Testing
    test_data = Data(bert_tokenizer=bert_tokenizer)
    test_samples = test_data.load_from_file(config.test_data_file)
    test_data.data_to_bert_input(test_samples)

    trainee.test(data=test_data)



