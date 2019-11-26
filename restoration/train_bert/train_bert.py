import logging
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.nn import DataParallel
from restoration.train_bert.model import FeedForwardNeuralNetwork
from restoration.data_util import config

logger = logging.getLogger("restoration")


class Trainee(object):
    def __init__(self, bert_model, no_cuda=False):
        super().__init__()
        self.device, self.n_gpu = self._check_device(no_cuda=no_cuda)
        self.bert_model = bert_model.to(self.device)
        self.ffnn_model = FeedForwardNeuralNetwork({
            "class-number": 2,
            "hidden-dimension": 768,
            "dropout-rate": 0.5
        })
        self.ffnn_model = self.ffnn_model.to(self.device)
        self._check_gpu_parallel()
        self._memory_monitor("Model Loaded")
        self.set_seed(1)

    def _check_device(self, no_cuda):
        if no_cuda:
            return torch.device("cpu"), -1
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

    def _memory_monitor(self, stage_name):
        if str(self.device) != "cuda":
            return
        logger.debug(" -*- Memory Monitor -*- ")
        logger.debug(" Stage: {}".format(stage_name))
        for i in range(self.n_gpu):
            logger.debug("[cuda {}] - Peak  Memory  Usage   {:.3f} M".format(i, torch.cuda.max_memory_allocated(device=i) / (1024*1024)))
            logger.debug("[cuda {}] - Current Memory Usage  {:.3f} M".format(i, torch.cuda.memory_allocated(device=i) / (1024*1024)))

    def save_stats(self, save_dir, filename, epoch, num_of_batch, cnt, value):
        path_to_file = os.path.join(save_dir, filename)
        if filename and save_dir:
            if not os.path.exists(path_to_file):
                with open(path_to_file, 'w', encoding='utf8') as file:
                    file.write("{}\t{}\t{}\t{}".format(epoch, num_of_batch, cnt, value))
                    file.write("\n")
            else:
                with open(path_to_file, 'a', encoding='utf8') as file:
                    file.write("{}\t{}\t{}\t{}".format(epoch, num_of_batch, cnt, value))
                    file.write("\n")

    def train(self, data, fine_tune=False, save_dir=None, backup=False):
        # For consistent model
        self.set_seed(1)

        # Initialize path
        if save_dir:
            if not os.path.exists(save_dir):
                raise FileNotFoundError("save_dir is specified but not found: {}".format(save_dir))

        # Initialize data loader
        data_loader = DataLoader(data.get_dataset(),
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=8,
                                 drop_last=True)

        # Initialize optimizers of BERT and FFNN
        if fine_tune:
            logger.info(" 😳 Parameters of BERT will be updated! 😳 ")
            bert_parameters = [p for n, p in list(self.bert_model.named_parameters())]
            classifier_parameters = bert_parameters + list(self.ffnn_model.parameters())
        else:
            logger.info(" 😟 Parameters of BERT will 'NOT' be updated! 😟 ")
            classifier_parameters = list(self.ffnn_model.parameters())
        optimizer = torch.optim.Adam(classifier_parameters, lr=config.lr)

        # Confusion Matrix
        cm = np.array([0, 0, 0, 0])  # tp, tn, fp, fn
        best_fscore = 0.0  # used for saving model

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

                    # Merge batch-size and doc-size together (confix.batch_size * config.max_len)
                    cls_feature = encoded_layers.view(-1, 768)

                    tag = self.ffnn_model(cls_feature)

                    # Recover the shape and swap axis 1,2 to fit the format of cross entropy
                    tag = tag.view(config.batch_size, config.max_len, -1)
                    _tag = tag.permute(0, 2, 1)

                    tag_loss = F.cross_entropy(_tag, target, reduction="sum")
                    tag_loss /= config.batch_size

                    _tag = tag.view(-1, 2)
                    _target = target.view(-1, 1)
                    cm += self._cm(_tag, _target)

                    if cnt % log_interval == 0:
                        # Inspect Data
                        logger.info(" === Answer === ")
                        data.tag_to_word(feature[0], target[0])
                        logger.info(" ===  Pred  === ")
                        values, index = torch.max(tag[0], dim=1)
                        data.tag_to_word(feature[0], index)

                        # Memory Monitor
                        self._memory_monitor("[STEP {}/{}] - Training".format(cnt, num_of_batch))

                        # Loss output
                        # tag_loss = tag_loss / config.max_len
                        tqdm.write("[{}/{}] LOSS = {:.3f}".format(cnt, num_of_batch, tag_loss.item()))

                        # Calculate scores including f1score, accuracy, precision, recall
                        scores = self._score(cm)
                        tqdm.write("F1 score   : {:.3f}".format(scores[0]))
                        tqdm.write("Accuracy   : {:.3f}".format(scores[1]))
                        tqdm.write("Precision  : {:.3f}".format(scores[2]))
                        tqdm.write("Recall     : {:.3f}".format(scores[3]))

                        # Saving to file
                        self.save_stats(save_dir, "f1_score.txt", epoch, num_of_batch, cnt, scores[0])
                        self.save_stats(save_dir, "accuracy.txt", epoch, num_of_batch, cnt, scores[1])
                        self.save_stats(save_dir, "precision.txt", epoch, num_of_batch, cnt, scores[2])
                        self.save_stats(save_dir, "recall.txt", epoch, num_of_batch, cnt, scores[3])

                        # Save Model
                        if backup:
                            if scores[0] > best_fscore:
                                logger.info("[Epoch {}][Step {}/{}] Save model to {}".format(epoch, cnt, num_of_batch,
                                                                                             config.trained_ffnn_file))
                                self.save_model(config.trained_ffnn_file)
                                logger.info("[Epoch {}][Step {}/{}] Save bert to {}".format(epoch, cnt, num_of_batch,
                                                                                            config.trained_bert_file))
                                self.save_bert(config.trained_bert_file)
                                best_fscore = scores[0]
                            else:
                                logger.info("[Epoch {}][Step {}/{}] Models are not saved! F1 score {} is lower than {}".format(epoch, cnt, num_of_batch, scores[0], best_fscore))

                        # Reset all values of cm
                        cm = np.array([0, 0, 0, 0])

                    self.save_stats(save_dir, "loss.txt", epoch, num_of_batch, cnt, tag_loss.item())

                    tag_loss.backward()

                    # Step
                    optimizer.step()
                    cnt += 1
                    pbar.update(1)

    def predict(self, feature, segments_tensor, attns_tensor):
        """
        Args:
            feature (str): A sentence which consists of 5 utterance in this form:
            "sent1 sent2 sent3 sent4 sent5"
        Returns:
        """
        # Set seed
        # self.set_seed(1)
        # Checkout device
        feature = feature.to(self.device)
        segments_tensor = segments_tensor.to(self.device)
        attns_tensor = attns_tensor.to(self.device)

        # Unsqueeze to make batch size 1
        feature = feature.unsqueeze(0)
        segments_tensor = segments_tensor.unsqueeze(0)
        attns_tensor = attns_tensor.unsqueeze(0)

        # consistent
        self.set_seed(1)

        # BERT Part
        self.bert_model.eval()

        encoded_layers, _ = self.bert_model(feature,
                                            attention_mask=attns_tensor,
                                            token_type_ids=segments_tensor)
        encoded_layers.to(self.device)

        # Feed Forward NN Part
        self.ffnn_model.eval()

        # Merge batch-size and doc-size together (confix.batch_size * config.max_len)
        cls_feature = encoded_layers.view(-1, 768)

        tag = self.ffnn_model(cls_feature)

        # Recover the shape
        tag = tag.view(-1, 2)
        values, index = torch.max(tag, dim=1)
        return index

    def set_seed(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def test(self, data, save_dir=None):
        # Set seed
        self.set_seed(1)
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
        bert_config = os.path.join(dir_name, "config.json")
        model_to_save = self.bert_model.module if hasattr(self.bert_model, 'module') else self.bert_model

        to_save_dict = model_to_save.state_dict()
        to_save_with_prefix = {}
        for key, value in to_save_dict.items():
            to_save_with_prefix['bert.' + key] = value
        torch.save(to_save_with_prefix, bert_file)
        with open(bert_config, 'w') as f:
            f.write(model_to_save.config.to_json_string())
