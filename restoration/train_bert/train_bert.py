import logging
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.nn import DataParallel
from transformers import BertModel, BertTokenizer
from restoration.train_bert.model import FeedForwardNeuralNetwork
from restoration.train_bert.dataset import Data
from restoration.data_util import config

logger = logging.getLogger("restoration")


class Trainee(object):
    def __init__(self, bert_model):
        super().__init__()
        self.device, self.n_gpu = self._check_device()
        self.bert_model = bert_model.to(self.device)
        self.ffnn_model = FeedForwardNeuralNetwork({
            "class-number": 2,
            "hidden-dimension": 768,
            "dropout-rate": 0.5
        }).to(self.device)
        self._check_gpu_parallel()

    def _check_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "cuda":
            n_gpu = torch.cuda.device_count()
            return device, n_gpu
        return device, -1

    def _check_gpu_parallel(self):
        if self.n_gpu > 1:
            logger.info(" 🧙‍ Using GPU Paralleling : {n_gpu} GPUs 🧙‍".format(n_gpu=self.n_gpu))
            self.bert_model = DataParallel(self.bert_model, device_ids=list(range(self.n_gpu)), dim=0)
            self.ffnn_model = DataParallel(self.ffnn_model, device_ids=list(range(self.n_gpu)), dim=0)

    def train(self, data, fine_tune=False, save_dir=None, backup=False):
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
            classifier_parameters = bert_parameters + list(self.ffnn_model.parameters())
        else:
            classifier_parameters = list(self.ffnn_model.parameters())
        optimizer = torch.optim.Adam(classifier_parameters, lr=config.lr)

        # Confusion Matrix
        cm = np.array([0, 0, 0, 0])  # tp, tn, fp, fn

        # Per epoch 
        for epoch in range(config.total_epochs):
            num_of_batch = len(data_loader.dataset) // config.batch_size

            # Logging interval for loss, cm, ...
            log_interval = num_of_batch // config.log_time_ratio
            log_interval = log_interval if log_interval != 0 else 1

            # Counter of step
            cnt = 1

            # Per batch
            with tqdm(total=num_of_batch) as pbar:
                for feature, target, segments_tensors, attns_tensors in data_loader:
                    # Checkout device
                    feature, target = feature.to(self.device), target.to(self.device)
                    segments_tensors = segments_tensors.to(self.device)
                    attns_tensors = attns_tensors.to(self.device)

                    optimizer.zero_grad()

                    # BERT part
                    if fine_tune:
                        self.bert_model.train()
                    else:
                        self.bert_model.eval()
                    encoded_layers, _ = self.bert_model(feature,
                                                        attention_mask=attns_tensors,
                                                        token_type_ids=segments_tensors)
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
                        cm += self._cm(tag, target[:, idx])

                    if cnt % log_interval == 0:
                        # Save Model
                        if backup:
                            logger.info("[Epoch {}][Step {}/{}] Save model to {}".format(epoch, cnt, num_of_batch,
                                                                                         config.trained_model_file))
                            self.save_model(config.trained_model_file)
                            logger.info("[Epoch {}][Step {}/{}] Save bert to {}".format(epoch, cnt, num_of_batch,
                                                                                        config.trained_bert_file))
                            self.save_bert(config.trained_bert_file)

                        # Loss output
                        # tag_loss = tag_loss / config.max_len
                        tqdm.write("[{}/{}] LOSS = {:.3f}".format(cnt, num_of_batch, tag_loss.item()))

                        # Calculate scores including f1score, accuracy, precision, recall
                        scores = self._score(cm)
                        tqdm.write("F1 score   : {:.3f}".format(scores[0]))
                        tqdm.write("Accuracy   : {:.3f}".format(scores[1]))
                        tqdm.write("Precision  : {:.3f}".format(scores[2]))
                        tqdm.write("Recall     : {:.3f}".format(scores[3]))

                        # Reset all values of cm
                        cm = np.array([0, 0, 0, 0])

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

        # Confusion Matrix
        cm = np.array([0, 0, 0, 0])  # tp, tn, fp, fn

        with tqdm(total=num_of_batch) as pbar:

            # Logging interval for loss, cm, ...
            log_interval = num_of_batch // config.log_time_ratio
            log_interval = log_interval if log_interval != 0 else 1

            for feature, target, segments_tensors, attns_tensors in data_loader:
                # Checkout device
                feature, target = feature.to(self.device), target.to(self.device)
                segments_tensors = segments_tensors.to(self.device)
                attns_tensors = attns_tensors.to(self.device)

                # BERT part
                self.bert_model.eval()
                encoded_layers, _ = self.bert_model(feature,
                                                    attention_mask=attns_tensors,
                                                    token_type_ids=segments_tensors)
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
                    cm += self._cm(tag, target[:, idx])

                # Loss output
                # tag_loss = tag_loss / config.max_len
                if cnt % log_interval == 0:
                    tqdm.write("[{}/{}] LOSS = {:.3f}".format(cnt, num_of_batch, tag_loss.item()))

                if loss_stats and save_dir:
                    if not os.path.exists(loss_stats):
                        with open(loss_stats, 'w', encoding='utf8') as file:
                            file.write("{}\t{}\t{}".format(num_of_batch, cnt, tag_loss.item()))
                            file.write("\n")
                    else:
                        with open(loss_stats, 'a', encoding='utf8') as file:
                            file.write("{}\t{}\t{}".format(num_of_batch, cnt, tag_loss.item()))
                            file.write("\n")

                # Calculate scores including f1score, accuracy, precision, recall
                if cnt % log_interval == 0:
                    scores = self._score(cm)
                    tqdm.write("F1 score   : {:.3f}".format(scores[0]))
                    tqdm.write("Accuracy   : {:.3f}".format(scores[1]))
                    tqdm.write("Precision  : {:.3f}".format(scores[2]))
                    tqdm.write("Recall     : {:.3f}".format(scores[3]))
                    tqdm.write("")

                    # Reset all values of cm
                    cm = np.array([0, 0, 0, 0])

                # Step
                cnt += 1
                pbar.update(1)

    def _cm(self, predicted_tag, gold_tag):
        """
        Compute confusion matrix
        Args:
            predicted_tag (torch.tensor): shape = (batch_size, class_number)
            gold_tag (torch.,tensor): (batch_size, class_number)
        Returns:
            confusion_matrix (numpy.array): [tp, tn, fp, fn]
        """
        values, index = torch.max(predicted_tag, dim=1)
        pred_tag, gold_tag = index.to("cpu").numpy(), gold_tag.to("cpu").numpy()

        assert pred_tag.size == gold_tag.size

        # Calculate tp, tn, fp, fn
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(pred_tag.size):
            if pred_tag[i] == gold_tag[i]:
                if pred_tag[i] == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if pred_tag[i] == 1 and gold_tag[i] == 0:
                    fp += 1
                else:
                    fn += 1

        return np.array([tp, tn, fp, fn])

    def _score(self, confusion_matrix):
        """
        Args:
            confusion_matrix (numpy.array): [tp, tn, fp, fn]
        Raises:
            ValueError: zero division
        Returns:
            score (numpy.array): [f1score, accuracy, precision, recall]
        """
        tp, tn, fp, fn = confusion_matrix

        if tp + fn == 0:
            logger.warning("There are no positive examples! No way to get scores")
            return np.array([0, 0, 0, 0])
        if tp + fp == 0:
            logger.warning("There are no samples predicted as positive! Either model is broken or \
            given data is not balanced")
            return np.array([0, 0, 0, 0])
        if tp == 0:
            logger.warning("There are no true positive! Either model is broken or given data is not balanced")
            return np.array([0, 0, 0, 0])

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        f1score = 2 * precision * recall / (precision + recall)

        return np.array([f1score, accuracy, precision, recall])

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
    # Logging Configuration
    logging.basicConfig(format="%(asctime)s [%(threadName)s-%(process)d] %(levelname)-5s %(module)s - %(message)s",
                        level=logging.INFO)

    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_file)
    bert_model = BertModel.from_pretrained(config.bert_file)

    # Data
    data = Data(bert_tokenizer=bert_tokenizer)
    samples = data.load_from_file(config.data_file)
    data.data_to_bert_input(samples)

    # Training
    trainee = Trainee(bert_model=bert_model)
    trainee.train(data=data, fine_tune=False, backup=True)

    # Testing
    # test_data = Data(bert_tokenizer=bert_tokenizer)
    # test_samples = test_data.load_from_file(config.test_data_file)
    # test_data.data_to_bert_input(test_samples)
    #
    # trainee.test(data=test_data)



