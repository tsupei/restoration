import os
import json
import yaml


class Parameter(object):
    def __init__(self, epochs, bsz, lr, bert, ffnn):
        self.bsz = bsz
        self.lr = lr
        self.epochs = epochs
        self.bert = bert
        self.ffnn = ffnn

    @classmethod
    def from_file(cls, filename):
        # .json
        # .yaml
        # .txt
        if not os.path.exists(filename):
            raise FileNotFoundError("{} is not found".format(filename))
        if "." not in filename:
            raise ValueError("{} is not a valid filename".format(filename))

        ftype = filename.rsplit('.', 1)[1]

        if ftype == "json":
            with open(filename, "r", encoding="utf8") as file:
                params = json.loads(file.read())
                return Parameter(epochs=params["epochs"],
                                 bsz=params["bsz"],
                                 lr=params["lr"],
                                 bert=params["bert"],
                                 ffnn=params["ffnn"])
        elif ftype == "yaml":
            with open(filename) as stream:
                params = yaml.safe_load(stream)
                return Parameter(epochs=params["epochs"],
                                 bsz=params["bsz"],
                                 lr=params["lr"],
                                 bert=params["bert"],
                                 ffnn=params["ffnn"])
        else:
            ValueError("{} is not supported".format(ftype))
