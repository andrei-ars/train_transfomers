import numpy as np
import pandas as pd

# Creating train_df  and eval_df for demonstration
train_data = [
    [0, "Simple", "B-MISC"],
    [0, "Transformers", "I-MISC"],
    [0, "started", "O"],
    [0, "with", "O"],
    [0, "text", "O"],
    [0, "classification", "B-MISC"],
    [1, "Simple", "B-MISC"],
    [1, "Transformers", "I-MISC"],
    [1, "can", "O"],
    [1, "now", "O"],
    [1, "perform", "O"],
    [1, "NER", "B-MISC"],
]
train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])

eval_data = [
    [0, "Simple", "B-MISC"],
    [0, "Transformers", "I-MISC"],
    [0, "was", "O"],
    [0, "built", "O"],
    [0, "for", "O"],
    [0, "text", "O"],
    [0, "classification", "B-MISC"],
    [1, "Simple", "B-MISC"],
    [1, "Transformers", "I-MISC"],
    [1, "then", "O"],
    [1, "expanded", "O"],
    [1, "to", "O"],
    [1, "perform", "O"],
    [1, "NER", "B-MISC"],
]
eval_df = pd.DataFrame(eval_data, columns=["sentence_id", "words", "labels"])

