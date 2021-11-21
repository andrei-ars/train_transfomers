import os
import sys
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from simpletransformers.ner import NERModel
from transformers import BertTokenizer #, BertModel, BertForMaskedLM
from sklearn.metrics import f1_score, accuracy_score

from confidence import calc_confidence


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


class NerModel:
    def __init__(self, modelname="", dataset=None, use_saved_model=False, 
                    input_dir=None, output_dir=None):
        
        #pretrained_model_name = "lm_outputs_test/from_scratch/best_model"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.dataset = dataset
        #labels_list = ["O", "B-ACT",  "I-ACT", "B-OBJ", "I-OBJ", "B-VAL", "I-VAL", "B-VAR", "I-VAR"]
        #labels_list = dataset.get_labels_list()
        
        labels_list = dataset['labels_list']
        #labels_list = ['O', 'B-ACT', 'I-ACT', 'B-OBJ', 'I-OBJ', 'B-CNT', 'I-CNT', 
        #    'B-OPE', 'I-OPE', 'B-ORD', 'B-PRE', 'I-PRE', 'B-TYP', 
        #    'B-VAL', 'I-VAL', 'B-ATT', 'I-ATT', 'B-VAR', 'I-VAR']

        #output_dir = "outputs_{}".format(modelname)
        os.system("{} -rf".format(output_dir))

        use_cuda = torch.cuda.is_available()

        # Create a NERModel
        model_args = {
            'labels_list': labels_list,

            'output_dir': output_dir,
            'overwrite_output_dir': True,
            'reprocess_input_data': True,
            
            'save_eval_checkpoints': False,
            'save_steps': -1,
            'save_model_every_epoch': False,
            #'no_save' : True,
            #'no_cache': True,
            'evaluate_during_training' : True,
            
            'num_train_epochs': 15, # 5
            'train_batch_size': 10, # 10   (<=10 for bert, <=5 for longformer)
            'eval_batch_size' : 10,
            'max_seq_length': 128,  # default 128
            'gradient_accumulation_steps': 8,
            'learning_rate': 0.0001, # default 4e-5; a good value is 0.0001 for albert

            #'max_position_embeddings': 64,
        }

        #self.model = NERModel("bert", pretrained_model_name, use_cuda=False, args=model_args)
        #self.model = NERModel("bert", "bert-base-uncased", use_cuda=False, args=model_args)
        #self.model = NERModel("longformer", "allenai/longformer-base-4096", use_cuda=False, args=model_args)
        #self.model = NERModel("longformer", pretrained_model_name, use_cuda=False, args=model_args)
        #self.model = NERModel("xlmroberta", "xlm-roberta-base", use_cuda=False, args=model_args)
        #self.model = NERModel("albert", "albert-base-v2", use_cuda=False, args=model_args)
        #self.model = NERModel("electra", 'google/electra-small-generator', use_cuda=False, args=model_args)
        #self.model = NERModel("layoutlm", 'microsoft/layoutlm-base-uncased', use_cuda=False, args=model_args)
        #self.model = NERModel("distilbert", "distilbert-base-cased-distilled-squad", use_cuda=False, args=model_args)

        #model_type, english_model_name  = "longformer", "allenai/longformer-base-4096"
        #model_type, english_model_name  = "mpnet", "microsoft/mpnet-base"
        #model_type, english_model_name  = "electra", "google/electra-small-discriminator"
        #model_type, english_model_name  = "squeezebert", "squeezebert/squeezebert-uncased"
        #model_type, english_model_name  = "albert", "albert-base-v2"
        #model_type, english_model_name  = "xlmroberta", "xlm-roberta-base"
        model_type, english_model_name  = "roberta", "distilroberta-base"
        #model_type, english_model_name  = "bert", "bert-base-uncased"
        #model_type, english_model_name  = "distilbert", "distilbert-base-uncased"

        if input_dir:
            # Use a previously trained model (on NER or LM tasks)
            self.model = NERModel(model_type, input_dir, use_cuda=use_cuda, args=model_args)
        else:
            # Use a pre-trained (English) model
            self.model = NERModel(model_type, english_model_name, use_cuda=use_cuda, args=model_args) # force_download=True

        """
        if use_saved_model:
            if path:
                # Use a model located in a given folder
                self.model = NERModel("longformer", path, use_cuda=False, args=model_args)
            else:
                # Use a previously trained model (on NER or LM tasks)
                self.model = NERModel("longformer", output_dir, use_cuda=False, args=model_args)
        else:
            # Use a pre-trained (English) model
            self.model = NERModel("longformer", "allenai/longformer-base-4096", use_cuda=False, args=model_args)
        """


        """
        if use_saved_model:
            self.model = NERModel("bert", output_dir, use_cuda=False, args=model_args)
        else:
            self.model = NERModel("bert", pretrained_model_name, use_cuda=False, args=model_args)
            # args={"overwrite_output_dir": True, "reprocess_input_data": True}
        """

        self.model_info = {'model_type':model_type, 'english_model_name':english_model_name, 'input_dir':input_dir}

    def train(self):
        # # Train the model
        if self.dataset:
            global_step, training_details = self.model.train_model(self.dataset['train'], eval_data=self.dataset['val'])
        else:
            raise Exception("dataset is None")

        print("global_step:", global_step)
        print("training_details:", training_details)
        #training_details: {'global_step': [4], 'precision': [0.6987951807228916], 'recall': [0.402777777777777
        #8], 'f1_score': [0.5110132158590308], 'train_loss': [0.41127926111221313], 'eval_loss': [0.63655577600
        #00229]}
        # it contains f1_score only for the validation dataset
        return training_details

    def eval(self):
        # # Evaluate the model
        if self.dataset:
            res_train, model_outputs, predictions = self.model.eval_model(
                                                    self.dataset['train'])
            res_val, model_outputs, predictions = self.model.eval_model(
                                                    self.dataset['val'])
            print("Evaluation")
            #print("On train data:", result)
            #{'eval_loss': 0.8920, 'precision': 0.0833, 'recall': 0.027, 'f1_score': 0.0416}
            print("train loss: {:.3f}; prec/recall/f1: {:.3f}/{:.3f}/{:.3f}".format(
                res_train['eval_loss'], res_train['precision'], res_train['recall'], res_train['f1_score']))
            #print("On validation data:", result)
            print("valid loss: {:.3f}; prec/recall/f1: {:.3f}/{:.3f}/{:.3f}".format(
                res_val['eval_loss'], res_val['precision'], res_val['recall'], res_val['f1_score']))
            print("Summary. Loss (train/val): {:.3f}/{:.3f}, F1: {:.3f}/{:.3f}".format(
                res_train['eval_loss'], res_val['eval_loss'], res_train['f1_score'], res_val['f1_score']))
        else:
            raise Exception("dataset is None")

        print("model_info:", self.model_info)

        return res_val

    def test(self):
        sentence_id = self.dataset['test']['sentence_id']
        words = self.dataset['test']['words']
        labels = self.dataset['test']['labels']
        
        prev_id = 0
        s_words = []
        s_labels = []
        samples = []

        for i in range(len(sentence_id)):
            s_id = sentence_id[i]
            word = words[i]
            label = labels[i]

            if s_id != prev_id:
                sentence = " ".join(s_words)
                #print("sentence id={}: {}".format(prev_id, sentence))
                samples.append({'text': sentence, 'tokens': s_words, 'labels': s_labels})
                #print("s_labels: {}".format(s_labels))
                s_words = []
                s_labels = []
                prev_id = s_id

            s_words.append(words[i])
            s_labels.append(labels[i])
            #print("i={}, word={}, label={}".format(s_id, word, label))

        sentence = " ".join(s_words)
        #print("sentence id={}: {}".format(prev_id, sentence))
        samples.append({'text': sentence, 'tokens': s_words, 'labels': s_labels})

        texts = [sample['text'] for sample in samples]
        predictions, raw_outputs = self.model.predict(texts)
        #print(predictions)

        acc_list = []
        success_list = []

        # More detailed preditctions
        for i, (preds, raw_outs) in enumerate(zip(predictions, raw_outputs)):
            print()
            print("text: ", texts[i])
            #print("\npreds: ", preds)
            pred_labels = [list(t.values())[0] for t in preds]
            print("pred_labels: ", pred_labels)
            true_labels = samples[i]['labels']
            print("true_labels: ", true_labels)
            #print("raw_outs: ", raw_outs)
            
            if len(true_labels) != len(pred_labels):
                raise Exception("len(true_labels) != len(pred_labels)")
            comp = [true_labels[i] == pred_labels[i] for i in range(len(pred_labels))]
            acc1sentence = np.mean(comp)
            print("acc={:.3f}".format(acc1sentence))
            acc_list.append(acc1sentence)
            success = 1 if acc1sentence == 1.0 else 0
            success_list.append(success)

        avg_acc = np.mean(acc_list)
        avg_success = np.mean(success_list)

        return {'avg_acc': avg_acc, 'avg_success': avg_success}

            #for pred, out in zip(preds, outs):
                #print("pred:", pred)
                #print("out:", out)
                #key = list(pred.keys())[0]
                #new_out = out[key]
                #preds = list(softmax(np.mean(new_out, axis=0)))
                #print(key, pred[key], preds[np.argmax(preds)], preds)


    def simple_test(self):
        # Predictions on arbitary text strings
        sentences = ["Some arbitary sentence", "Simple Transformers sentence"]
        predictions, raw_outputs = self.model.predict(sentences)
        print(predictions)

        # More detailed preditctions
        for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
            print("\n___________________________")
            print("Sentence: ", sentences[n])
            for pred, out in zip(preds, outs):
                key = list(pred.keys())[0]
                new_out = out[key]
                preds = list(softmax(np.mean(new_out, axis=0)))
                print(key, pred[key], preds[np.argmax(preds)], preds)

    
    def predict(self, sentences):
        predictions, raw_outputs = self.model.predict(sentences)
        #tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        #predictions, raw_outputs = self.model.predict(tokenized_sentences, split_on_space=False)
        return predictions

    def raw_predict(self, sentences):
        predictions, raw_outputs = self.model.predict(sentences)
        #print("raw_outputs:", raw_outputs)
        #print(self.model.args.labels_list)
        labels_list = self.model.args.labels_list
        confidences = [calc_confidence(raw_output, labels_list) for raw_output in raw_outputs]
        #print("confidence:", confidence)
        return {'predictions': predictions, 'raw_outputs': raw_outputs, 'confidences': confidences}
        """
          labels_list=['O', 'B-ACT', 'B-CNT', 'B-OBJ', 'B-OPE', 'B-ORD', 'B-PRE', 'B-TYP', 
            'B-VAL', 'I-ACT', 'I-CNT', 'I-OBJ', 'I-OPE', 'I-PRE']

        Click on Basket
        {'predictions': [[{'Click': 'B-ACT'}, {'on': 'I-ACT'}, {'Basket': 'B-OBJ'}]], 
        'raw_outputs': [[{'Click': [[0.9166969, 7.8369393, 0.31039014, -0.60283166, -1.2205212, -1.0528294, -0.57920927, -1.8390691, -0.7053572, 0.72872484, -1.2057197, -1.1906811, -0.2462096, -0.73678666]]}, 
        {'on': [[2.864785, 0.81569016, -2.4111693, -1.753845, -1.3591471, -0.45600742, 0.22175379, -0.5122926, -0.43606478, 6.4333878, -1.3908641, -0.48224172, -0.9318897, -1.1649382]]}, 
        {'Basket': [[3.537952, -0.11874729, -0.6143042, 4.150231, -1.6462238, 0.49634176, -1.2271496, -1.7730664, 0.10150815, 0.36229157, -1.1427803, 1.7072755, -0.07331692, -0.8622093], 
        [0.2326435, -0.2758199, -0.67160505, 5.107534, -1.1315546, -0.060734212, -1.6606357, -0.8778804, -0.74439603, -0.7747257, -1.1741363, 3.4446402, -0.66995436, -0.360082]]}]], 
        'confidences': [[0.996, 0.9613, 0.5701]]}

        Click on OK button
        - 'confidences': [[0.9965, 0.9836, 0.9913, 0.8678]]

        Click on BOQ OK EOQ button
        """