import logging
from transformers import BertModel, BertTokenizer
from restoration.train_bert.dataset import Data
from restoration.train_bert.train_bert import Trainee
from restoration.data_util import config
from restoration.data_util.parameter import Parameter

# Logging Configuration
logging.basicConfig(format="%(asctime)s [%(threadName)s-%(process)d] %(levelname)-5s %(module)s - %(message)s",
                        level=logging.DEBUG)

LOSS_DIR = ""
PRETRAINED_BERT = ""
PRETRAINED_FFNN = ""
SAVE_BERT = ""
SAVE_FFNN = ""
DATA_FILE = ""


def train(pretrained_ffnn=None):
    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT)
    bert_model = BertModel.from_pretrained(PRETRAINED_BERT)

    # Parameter
    params = Parameter(epochs=5, bsz=3, lr=0.00001, bert=SAVE_BERT, ffnn=SAVE_FFNN)

    # Data
    data = Data(bert_tokenizer=bert_tokenizer)
    samples = data.load_from_file(DATA_FILE)
    data.data_to_bert_input(samples[:20000])

    # Training
    trainee = Trainee(bert_model=bert_model)
    if pretrained_ffnn:
        trainee.load_model(PRETRAINED_FFNN)
    trainee.train(data=data, params=params, save_dir=LOSS_DIR, fine_tune=True, backup=True)


def predict(file=None, sent=None):
    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT)
    bert_model = BertModel.from_pretrained(PRETRAINED_BERT)

    # Data
    data = Data(bert_tokenizer=bert_tokenizer)

    # Invoke Trainee
    trainee = Trainee(bert_model=bert_model, no_cuda=True)
    trainee.load_model(PRETRAINED_FFNN)

    # Predict
    if sent:
        one_data = {
            "feature": sent
        }
        one_feature, one_segment, one_attn = data.one_data_to_bert_input(one_data)

        # DEBUG:
        logging.debug("Feature: {}".format(one_feature))
        logging.debug("Segment: {}".format(one_segment))
        logging.debug("Attent : {}".format(one_attn))

        tags = trainee.predict(one_feature, one_segment, one_attn)
        words = data.tag_to_word(one_feature, tags)

    if file:
        # Parameter
        params = Parameter(epochs=5, bsz=3, lr=0.00001, bert=SAVE_BERT, ffnn=SAVE_FFNN)

        samples = data.load_from_file(file)
        data.data_to_bert_input(samples)
        trainee.test(data=data, params=params, save_dir=LOSS_DIR)


if __name__ == "__main__":
    # train(True)
    predict(sent="什麼是心經啊 \
心經經文組織嚴密有序，分段闡述空的意義和層次 \
心經的緣由？ \
華嚴經是釋尊成道之後，首先宣說的經典 \
告訴我那是什麼")
# , file="/Users/admin/Practice/restoration/data/test_zh_tw.json")
