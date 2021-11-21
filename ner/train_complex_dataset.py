import os
import time
import numpy as np
import pandas as pd

from nermodel import NerModel
from nerdataset import NerPartDataset, NerDataset
from nerdataset import get_labels_list
from ner_slot_filling import split_token_tag #, ner_slot_filling
from ner_slot_filling import ner_slot_filling #, ner_slot_filling_compound

if __name__ == "__main__":

    mode = "train"
    #mode = "test"
    #mode = "infer"
    pretrained_type = "English"
    #pretrained_type = "LM"
    #pretrained_type = "continue"

    modelname = "nlp_complex_pre"
    #modelname = "nlp_complex_pre_table"
    #complex_dataset_names = ["table", "table_nq", "nlp_ext", "nlp_ext_nq"]
    complex_dataset_names = ["nlp_ext", "nlp_ext_nq", "compound", "compound_nq"]
    #complex_dataset_names = ["nlp_ext", "nlp_ext_nq", "compound", "compound_nq", "table", "table_nq", "nlp", "nlp_data"]
    #complex_dataset_names = ["nlp_ext_nq"]

    output_dir = "outputs_{}".format(modelname)

    """
    test_sentences = [
        "Double Click on a BOQ calendar EOQ from the list on the left side of the screen.",
        "Enter text into the BOQ password EOQ on the bottom left of the screen",
        "Double Click on a calendar from the list on the left side of the screen.",
        "Enter text into the password on the bottom left of the screen",
        "Enter text into the second name box on the bottom left of the screen",
        "Enter text into the Input name on the bottom left of the screen",
        "click on Find an Agent after Renters Insurance",
        "Hover over login after forgot your password?",
        "click on Get a Quote next to Motor Home Insurance",
        "Click on Lifestyle next to Bihar Election 2020",
        "click on Renters Insurance before Find an Agent",
        "click on Images next to Maps",
        "click on Simple Images next to Hello Maps",
        "click on Yes radio for Are you Hispanic or Latino?",
        "click on Letters where Numbers greater than 3",
        "Click on Manage External User button for Contact Detail",
        "Click on Black Where ends with c",
        "Navigate to leads page by clicking on Next step",
        "Change the page by clicking on Next button",
        "Extract information by clicking on Next button",
        ]
    """

    test_sentences = [
        "Double Click on a calendar from the list on the left side of the screen.",
        "Enter text into the password on the bottom left of the screen",
        "Enter text into the second name box on the bottom left of the screen",
        "Go out from the website by clicking on Log out from website",
        "Click on Basket in home window",
        "Click on Basket button in the left side of the screen",
        "Click on BOQ Basket EOQ button in the left side of the screen",
        "Double Click on a calendar from the list on the left side of the screen",
        "Click on calendar from the list on the left side of the screen",
        "Click on YYY from the list on the left side of the screen",
        "Double click on YYY aaa from the list on the left side of the screen",
        "enter username, password and click on Submit",
        "Enter in abstract, and click on Submit button",
        "Select document after clicking Browse",
        "Enter in BOQ abstract EOQ after clicking on Submit button",
        "Enter in big value after clicking on Submit button",
        "Enter in username box after clicking on Submit button",
        "Enter in abstract before clicking on Submit button",
        ]

    if mode == "train":
        dataset = NerDataset(complex_dataset_names).as_dict()
    else:
        dataset = NerDataset(complex_dataset_names).as_dict()
        #dataset = NerDataset(complex_dataset_names, labels_only=True).as_dict()
        #dataset['labels_list'] = get_labels_list("dataset/{}/tag.dict".format(modelname))

    if mode == "train":
        if pretrained_type == "LM":
            model = NerModel(modelname=modelname, dataset=dataset,
                            input_dir="lm_outputs_test/from_scratch/best_model",
                            output_dir=output_dir)
        elif pretrained_type == "continue":
            model = NerModel(modelname=modelname, dataset=dataset,
                            input_dir=output_dir,
                            output_dir=output_dir)
        elif pretrained_type == "English":
            model = NerModel(modelname=modelname, dataset=dataset,
                            output_dir=output_dir)

        training_details = model.train()
    
    elif mode in {"test", "infer"}:
        model = NerModel(modelname=modelname, dataset=dataset,
                            input_dir=output_dir,
                            output_dir=output_dir)
        #model = NerModel(modelname=modelname, dataset=dataset, use_saved_model=True,
        #                    path="outputs_{}".format(modelname))

    if mode in {"train", "test"}:
        print("\nMODEL.TEST:")
        test_results = model.test()
        model.eval()
        #model.eval()
        #predictions = model.predict(test_sentences)
        #for i in range(len(predictions)):
        #    text = test_sentences[i]
        #    print("text: {}\noutput: {}\n".format(text, predictions[i]))
        print("\navg acc={:.3f}".format(test_results['avg_acc']))
        print("avg success={:.3f}".format(test_results['avg_success']))

    if mode == "train":
        print("\ntraining_details:")
        print("global_step:", training_details['global_step'])
        train_loss = training_details['train_loss']
        val_loss = training_details['eval_loss']
        f1_score = training_details['f1_score']
        print("epoch | t_loss  v_loss | f1(val)")
        for i in range(len(train_loss)):
            print("{:2d} | {:.4f} {:.4f} | {:.3f}".\
                format(i, train_loss[i], val_loss[i], f1_score[i]))

    if mode in {"infer"}:
        print("Number of test_sentences:", len(test_sentences))
        result = model.raw_predict(test_sentences)
        time.sleep(5)
        predictions = result['predictions']
        raw_outputs = result['raw_outputs']
        for i in range(len(predictions)):
            text = test_sentences[i]
            print("text: {}\noutput: {}".format(text, predictions[i]))
            #tokens, tags = split_token_tag(predictions[i])
            #slots = ner_slot_filling(tokens, tags)
            #print("slots: {}\n".format(slots))

    print("\nManually input")
    while True:
        input_text = input("Input text: ")
        if input_text == 'q':
            break
        if "|" in input_text:
            input_text, input_data = input_text.split("|")

        predictions = model.predict([input_text])
        tokens, tags = split_token_tag(predictions[0])
        slots = ner_slot_filling(tokens, tags)
        #compound_result = ner_slot_filling_compound(tokens, tags)

        print("text: {}\noutput: {}".format(input_text, predictions[0]))
        #print("slots: {}\n".format(slots))
        print("compound_result: {}\n".format(compound_result))

        print("\n RAW:")
        results = model.raw_predict([input_text])
        predictions = results.get('predictions')
        raw_outputs = results.get('raw_outputs')
        confidences = results.get('confidences')
        print(predictions)
        for confidence in confidences:
            print("confidence min={:.4f}, mean={:.4f}".format(np.min(confidence), np.mean(confidence)))
