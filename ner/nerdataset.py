import numpy as np
import pandas as pd

DATASET_PATH = "ner_lm/dataset_ner"


def get_labels_list(filepath):
    labels_list = []
    with open(filepath) as fp:
        for line in fp:
            ls = line.split()
            if len(ls) == 2:
                labels_list.append(ls[0])
            else:
                logging.warning("bad format of the file {}".format(filepath))
    return labels_list


class NerPartDataset:
    def __init__(self, path):
        self.data = []

        with open(path) as fp:
            count = 0
            for linenum, line in enumerate(fp):
                ls = line.split()
                if len(ls) == 2:
                    token, tag = ls
                    if token == ".":
                        count += 1
                    else:
                        print("{}: {} - {}".format(count, token, tag))
                        self.data.append([count, token, tag])
                else:
                    print("len != 2")
                    print("path={}, linenum={}".format(path, linenum))
                    raise Exception("NerDataset.init: len != 2")

        #train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=["sentence_id", "words", "labels"])

    def as_list(self):
        return self.data


class NerDataset:
    def __init__(self, names, labels_only=False):

        path = DATASET_PATH

        labels = set()
        train_data = []
        val_data = []
        test_data = []
            
        for name in names:
            labels |= set(get_labels_list("{}/{}/tag.dict".format(path, name)))
            if not labels_only:
                train_data += NerPartDataset("{}/{}/train.txt".format(path, name)).as_list()
                val_data += NerPartDataset("{}/{}/valid.txt".format(path, name)).as_list()
                test_data += NerPartDataset("{}/{}/test.txt".format(path, name)).as_list()

        #labels.sort(key=lambda x: 100*len(x) + 1*ord(x[0]) + 10*(ord(x[2]) if len(x)>1 else 0) )
        # form as ['O', 'B-ACT', 'I-ACT', 'B-CNT', 'I-CNT'
        labels = list(labels)
        labels.sort()
        labels = labels[-1:] + labels[:-1]   # in form as ['O', 'B-ACT', 'B-CNT',...
        print("labels: {}".format(labels))

        self.dataset = {}
        self.dataset['labels_list'] = labels
        self.dataset['train'] = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])
        self.dataset['val'] = pd.DataFrame(val_data, columns=["sentence_id", "words", "labels"])
        self.dataset['test'] = pd.DataFrame(test_data, columns=["sentence_id", "words", "labels"])
        print("test dataset:", self.dataset['test'])

    def as_dict(self):
        return self.dataset
    #    dc = {'train': self.dataset.train,
    #          'val': self.dataset.val,
    #          'test': self.dataset.test,
    #            }



if __name__ == "__main__":

    modelname = "table_nq"

    #dataset = {}
    #dataset['labels_list'] = get_labels_list("dataset/{}/tag.dict".format(modelname))
    #print("labels_list: {}".format(dataset['labels_list']))
    #dataset['train'] = NerPartDataset("dataset/{}/train.txt".format(modelname)).to_dataframe()
    #dataset['val'] = NerPartDataset("dataset/{}/valid.txt".format(modelname)).to_dataframe()
    #dataset['test'] = NerPartDataset("dataset/{}/test.txt".format(modelname)).to_dataframe()
    #print(dataset)

    dataset = NerDataset(["table", "table_nq", "nlp_ext", "nlp_ext_nq"]).as_dict()
    #dataset = NerDataset(["nlp_ext_nq"]).as_dict()
    print(dataset)