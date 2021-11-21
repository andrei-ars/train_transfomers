import numpy as np
import pandas as pd

from nermodel import NerModel
from nerdataset import NerPartDataset, NerDataset
from nerdataset import get_labels_list


if __name__ == "__main__":

    mode = "train"
    #mode = "infer"

    #modelname = "table_nq"
    #modelname = "nlp_ext"
    modelname = "nlp_ext_nq"

    if modelname == "nlp_ext":
        test_sentences = [
            "Double Click on a BOQ calendar EOQ from the list on the left side of the screen.",
            "Enter text into the BOQ password EOQ on the bottom left of the screen"
            ]
    elif modelname == "nlp_ext_nq":
        test_sentences = [
            "Double Click on a calendar from the list on the left side of the screen.",
            "Enter text into the password on the bottom left of the screen",
            "Enter text into the second name box on the bottom left of the screen",
            "Log out from the website by clicking on Log out from website",
            "Click on Basket in home window",
            "Double Click on a calendar from the list on the left side of the screen",
            "Click on calendar from the list on the left side of the screen",
            "Click on YYY from the list on the left side of the screen",
            "Double press on YYY aaa from the list on the left side of the screen",
            ]
    elif modelname == "table_nq":
        test_sentences = [
            "click on Find an Agent after Renters Insurance",
            "Hover over login after forgot your password?",
            "click on Get a Quote next to Motor Home Insurance",
            "Click on Lifestyle next to Bihar Election 2020",
            "click on Renters Insurance before Find an Agent",
            "click on Images next to Maps",
            "click on Yes radio for Are you Hispanic or Latino?",
            "click on Letters where Numbers greater than 3",
            "Click on Manage External User button for Contact Detail",
            "Click on Black Where ends with c"
            ]

        #sentences = ["Click on the OK button", "Click on the BOQ OK EOQ button"]
        #sentences = ["enter in city textbox", "enter Choose a flavor", "enter name in the last name textbox"]
    else:
        raise Exception("wrong model name")


    #dataset = {}
    #dataset['labels_list'] = get_labels_list("dataset/{}/tag.dict".format(modelname))
    #print("labels_list: {}".format(dataset['labels_list']))
    
    if mode == "train":
        dataset = NerDataset([modelname]).as_dict()
    else:
        dataset = NerDataset([modelname], labels_only=True).as_dict()

    if mode == "train":
        #dataset['train'] = NerPartDataset("dataset/{}/train.txt".format(modelname)).to_dataframe()
        #dataset['val'] = NerPartDataset("dataset/{}/valid.txt".format(modelname)).to_dataframe()
        #dataset['test'] = NerPartDataset("dataset/{}/test.txt".format(modelname)).to_dataframe()
        #dataset = NerDataset([modelname]).as_dict()
        model = NerModel(modelname=modelname, dataset=dataset)
        model.train()
        model.eval()

    if mode in {"infer"}:
        model = NerModel(modelname=modelname, dataset=dataset, use_saved_model=True)

    if mode in {"train", "infer"}:
        predictions = model.predict(test_sentences)
        for i in range(len(predictions)):
            text = test_sentences[i]
            print("text: {}\noutput: {}\n".format(text, predictions[i]))

    print("\nManually input")
    while True:
        input_text = input("Input text: ")
        if input_text == 'q':
            break
        if "|" in input_text:
            input_text, input_data = input_text.split("|")

        result = model.predict([input_text])
        print("result:", result)