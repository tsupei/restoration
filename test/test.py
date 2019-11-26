import logging
from transformers import BertModel, BertTokenizer
from restoration.train_bert.dataset import Data
from restoration.train_bert.train_bert import Trainee
from restoration.data_util import config

# Logging Configuration
logging.basicConfig(format="%(asctime)s [%(threadName)s-%(process)d] %(levelname)-5s %(module)s - %(message)s",
                        level=logging.DEBUG)


def train(pretrained_ffnn=None):
    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained(config.pretrained_bert_file)
    bert_model = BertModel.from_pretrained(config.pretrained_bert_file)

    # Data
    data = Data(bert_tokenizer=bert_tokenizer)
    samples = data.load_from_file(config.data_file)
    data.data_to_bert_input(samples[:20000])

    # Training
    trainee = Trainee(bert_model=bert_model)
    if pretrained_ffnn:
        trainee.load_model(config.pretrained_ffnn_file)
    trainee.train(data=data, save_dir=config.loss_stats, fine_tune=True, backup=True)


def mini_train(mtz=5000):
    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained(config.pretrained_bert_file)
    bert_model = BertModel.from_pretrained(config.pretrained_bert_file)

    # Data
    data = Data(bert_tokenizer=bert_tokenizer)
    samples = data.load_from_file(config.data_file)
    t_mini_trains = len(samples) // mtz

    # Training
    trainee = Trainee(bert_model=bert_model)

    for i in range(t_mini_trains):
        data.data_to_bert_input(samples[i*mtz : (i+1)*mtz])
        trainee.train(data=data, fine_tune=True, backup=True)


def predict():
    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained(config.pretrained_bert_file)
    bert_model = BertModel.from_pretrained(config.pretrained_bert_file)

    # Data
    data = Data(bert_tokenizer=bert_tokenizer)

    # Invoke Trainee
    trainee = Trainee(bert_model=bert_model)
    trainee.load_model(config.pretrained_ffnn_file)

    # Predict
    one_data = {
        "feature": "明明沒有很愛很愛一個人卻一直想逼他做我男朋友 你想太多了呵呵 日貴妃好重口味 那只是一個夢而已重嗎 你喜歡"
    }
    one_feature, one_segment, one_attn = data.one_data_to_bert_input(one_data)

    # DEBUG: 
    logging.debug("Feature: {}".format(one_feature))
    logging.debug("Segment: {}".format(one_segment))
    logging.debug("Attent : {}".format(one_attn))

    tags = trainee.predict(one_feature, one_segment, one_attn)
    logging.debug("Predict: {}".format(tags))
    words = data.tag_to_word(one_feature, tags)
    logging.info("Predict Words: {}".format(words))


if __name__ == "__main__":
    # train(True)
    predict()
