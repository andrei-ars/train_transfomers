"""
Results:
acc = 0.7188  [23/32]

"""
import os
import sys
import json
import pandas as pd
import random
import logging

from simpletransformers.classification import ClassificationModel


class ActionDataset():

    def __init__(self, data_path, data_format="json"):

        if data_format == "json":
            self.data = self.load_from_json(data_path)
        else:
            self.data = self.load_from_txt(data_path)



    def load_from_json(self, data_path):

        json_data ={}

        for mode in ['train', 'valid']:
            json_path = os.path.join(data_path, "{}.json".format(mode))
            print("Load dataset from {}".format(json_path))
            with open(json_path) as fp:
                json_data[mode] = json.load(fp)

        labels_set = {x['label'] for x in json_data['train']}
        labels = list(labels_set)
        labels.sort()
        self.index_to_label = {index: label for index, label in enumerate(labels)}
        self.label_to_index = {label: index for index, label in enumerate(labels)}


        data = {}
        for mode in ['train', 'valid']:
            data[mode] = [(x['text'], self.label_to_index[x['label']]) for x in json_data[mode]]
            print("mode {}: size={}".format(mode, len(data[mode])))

        return data


    def load_from_txt(self, data_path):

        line_by_line = True
        if os.path.isdir(data_path):
            if line_by_line:
                data = self.load_docs_line_by_line_from_files_in_directory(data_path)
            else:
                data = self.load_docs_from_directory(data_path)
        elif os.path.isfile(data_path):
            data = self.load_docs_from_file(data_path)
        else:
            raise Exception("data_path {} is not a file or a directory.".format(data_path))

        #data = [[sample[0], sample[2]] for sample in data]
        data = list(map(lambda x: (x[0], x[2]), data))
        # We will get:
        # [['Click zzdo .', 22], ['fill somedata .', 18], ['verify last BOQ zzxu zzpu EOQ .', 16],...

        random.shuffle(data)
        data_size = len(data)
        train_part = 0.9
        train_size = int(train_part * data_size)
        train_data = data[:train_size]
        val_data = data[train_size:]

        return {'train': train_data, 'valid': val_data}


    def load_docs_line_by_line_from_files_in_directory(self, data_path):
        """ Each line in each file in data_path is a document.
        """
        labels = [f for f in os.listdir(data_path)] #  if f.endswith('.txt')
        self.index_to_label = {index: label for index, label in enumerate(labels)}
        self.label_to_index = {label: index for index, label in enumerate(labels)}

        data = []
        self.text_number_to_label = {}
        for index, label in enumerate(labels):
            with open(data_path + '/' + label, 'r') as f:
                for line in f:
                    text = line.strip()
                    logging.debug('{}: {}, len={}'.format(index, label, len(text)))
                    data.append([text, label, index])
                    #self.text_number_to_label[index] = doc
        return data

    #def load_docs_from_directory(self, data_path):
    #    """ Each file in data_path is a whole document.
    #    """
    #    doc_labels = [f for f in os.listdir(data_path)] #  if f.endswith('.txt')
    #    data = []
    #    self.text_number_to_label = {}
    #    for i, doc in enumerate(doc_labels):
    #        with open(data_path + '/' + doc, 'r') as f:
    #            text = f.read()
    #            text = text.replace('\n', ' ')
    #            logging.debug('{}: {}, len={}'.format(i, doc, len(text)))
    #            data.append(text)
    #            self.text_number_to_label[i] = doc
    #    return data

    def load_docs_from_file(self, filepath):
        """ A single file filepath contains all docsIt should have the following form:
                    label ::: word word word
        """
        df = pd.read_csv(filepath, sep=' ::: ', names=['label','text'], engine='python')
        data = df.text
        self.text_number_to_label = {i:l for i,l in enumerate(list(df.label))}
        return data


    def __getitem__(self, index):
        if index == "train":
            return self.data['train']
        elif index == "valid":
            return self.data['valid']
        else:
            logging.warning("Bad index {}".format(index))
            return None


def train_model(model, dataset):

    # Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column is the text with type str, and the second column in the label with type int.
    #train_data = [
    #    ["Example sentence belonging to class 1", 1],
    #    ["Example sentence belonging to class 0", 0],
    #    ["Example eval senntence belonging to class 2", 2],
    #]
    train_df = pd.DataFrame(dataset['train'])

    #eval_data = [
    #    ["Example eval sentence belonging to class 1", 1],
    #    ["Example eval sentence belonging to class 0", 0],
    #    ["Example eval senntence belonging to class 2", 2],
    #]
    eval_df = pd.DataFrame(dataset['valid'])

    print("train_df:")
    print(train_df)
    print("eval_df:")
    print(eval_df)

    # Train the model
    print("Train the model")
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print("result:", result)
    print("model_outputs:", model_outputs)
    print("wrong_predictions:", wrong_predictions)
    return result, model_outputs, wrong_predictions



def test_model(model, index_to_label):

    samples = [
       ("Navigate to url www.hi.net/#/login .", "OPEN_WEBSITE"),

       ("Fill in text in BOQ Street Address line 1 EOQ .", "ENTER"),
       ('Set text in XPATH .', "ENTER"),

       ("choose XPATH New PAN Indian Citizen Form 49A .", "SELECT"),
       ('select BOQ Type EOQ .', "SELECT"),
       ('select Type .', "SELECT"),
       ('select .', "SELECT"),
       ('select BOQ Partnership Firm EOQ in BOQ PAN_APPLCNT_STATUS EOQ .', "SELECT"),

       ("Click XPATH .", "CLICK"),
       ("Press XPATH .",  "CLICK"),
       ('click BOQ Log In EOQ .',  "CLICK"),
       ('Click on first BOQ MoreVert EOQ .', "CLICK"),

       ("assert password after username .", "VERIFY"),
       ("assert username .", "VERIFY"),
       ('Verify text BOQ Disable Test Case EOQ .', "VERIFY"),

       ("Verify XPATH width is BOQ 235px EOQ .", "VERIFY_CSSPROP"),

       ("Verify XPATH is enabled .", "VERIFY_XPATH"),
       ("Verify XPATH is google .", "VERIFY_XPATH"),
       ('Verify XPATH contains Current or contains events .', "VERIFY_XPATH"),
       ('Verify XPATH contains Current or ends with events .', "VERIFY_XPATH"),
       ('Verify XPATH is Related Changes .', "VERIFY_XPATH"),
       ('verify XPATH is ername .', "VERIFY_XPATH"),
       
       ('scroll down .', "SCROLL_ACTION"),
       ('scroll up .', "SCROLL_ACTION"),
       
       ('Hit Enter .', "HIT"),
       ('Hit escape .', "HIT"),
       ('Hit spacebar .', "HIT"),
       ('hit tab .', "HIT"),
       ('hit up arrow key .', "HIT"),
       ('Begin block Block1 .', "BEGIN"),

       ('verify BOQ New Quote EOQ is visible on the page .', "VERIFY"),
       ('verify BOQ INSIDEQOUTES1 EOQ is visible on the page .', "VERIFY"),
    ]

    input_texts = list(map(lambda x: x[0], samples))
    true_labels = list(map(lambda x: x[1], samples))
    predictions, raw_outputs = model.predict(input_texts)
    predicted_labels = list(map(lambda index: index_to_label[index], predictions))
    print("predictions:", predictions)
    print("predicted_labels:", predicted_labels)

    count = 0
    for i in range(len(samples)):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        if true_label == predicted_label:
            print("+ {}".format(predicted_label))
            count += 1
        else:
            print("- wrong prediction: true={}, predicted={}".format(true_label, predicted_label))
            

    total_count = len(samples)
    acc = count / total_count
    print("\nacc = {:.4f}  [{}/{}]".format(acc, count, total_count))
    return acc


if __name__ == "__main__":

    mode = "train"
    #mode = "infer"

    #dataset = ActionDataset(data_path="./dataset_action")
    dataset = ActionDataset(data_path="./dataset/", data_format="json")

    print("train dataset size:", len(dataset['train']))
    print("val dataset size:", len(dataset['valid']))
    print("index_to_label:", dataset.index_to_label)
    print("Example of train data:")
    print(dataset['train'][:10])

    num_labels = len(dataset.index_to_label)
    print("num_labels:", num_labels)

    #sys.exit()

    # Create a ClassificationModel
    #  model_name is set to None to train a Language Model from scratch.
    if mode == "train":
        model = ClassificationModel(
            #model_type="bert", #"roberta", #"bert", 
            #model_name=None, # "bert-base-cased"; "xlnet-base-cased"
            model_type="roberta", #"roberta", #"bert", 
            model_name="distilroberta-base", # "bert-base-cased"; "xlnet-base-cased"
            num_labels=num_labels,
            args={"reprocess_input_data": True, 
                    "overwrite_output_dir": True,
                    'num_train_epochs': 1,   # 5
                    'train_batch_size': 32,  # 32 for bert (but >10 gives an error for longfomer)
                 },
            use_cuda=False
        )        
        train_model(model, dataset)
    
    elif mode == "infer":
        outputs_dir = "outputs"
        model = ClassificationModel(
            "roberta", 
            outputs_dir, 
            num_labels=num_labels,
            args={"reprocess_input_data": True, "overwrite_output_dir": False},
            use_cuda=False
        )        

    predictions, raw_outputs = model.predict(
            ["verify BOQ New Quote EOQ is visible on the page .",
             "Fill in text in BOQ Street Address line 1 EOQ ."]
            )
    print("predictions:", predictions)

    test_model(model, dataset.index_to_label)

