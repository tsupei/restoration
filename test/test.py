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
    data.data_to_bert_input(samples)

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
        "feature": "湖南有哪裏可以學習做甜品呀 我自己都想開在這裏 你也想開甜品店在長沙 長沙恐怕開不起來要回岳陽開 哈哈哈哈什麼時候開呀我一定去光顧"
    }
    one_feature, one_segment, one_attn = data.one_data_to_bert_input(one_data)
    tags = trainee.predict(one_feature, one_segment, one_attn)
    words = data.tag_to_word(one_feature, tags)
    logging.info("Predict Words: {}".format(words))


if __name__ == "__main__":
    train(pretrained_ffnn=True)
