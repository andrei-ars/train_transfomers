
import numpy as np
from mathfunctions import softmax_probabilities


def calc_confidence(raw_output, labels_list=None):
    """
    The input is a single result.
    The output is a list like [0.9983, 0.9614, 0.9978, 0.9953, 0.7372, 0.5323]
    """
    #print("raw_output:", raw_output)
    result = raw_output
    probs = []
    for dc in result:
        for key in dc:
            #print("dc:", dc)
            #print("key:", key)
            logits = dc[key]
            logit = logits[0]
            ps = softmax_probabilities(logit)
            max_ps = max(ps)
            index = np.argmax(ps)
            probs.append(max_ps)
            #print('max_ps:', max_ps)
            #print('index:', index)
            #if labels_list:
            #    print('tag:', labels_list[index])
    return probs


if __name__ == "__main__":
    raw_outputs = [[{'Click': [[0.9166969, 7.8369393, 0.31039014, -0.60283166, -1.2205212, -1.0528294, -0.57920927, -1.8390691, -0.7053572, 0.72872484, -1.2057197, -1.1906811, -0.2462096, -0.73678666]]}, {'on': [[2.864785, 0.81569016, -2.4111693, -1.753845, -1.3591471, -0.45600742, 0.22175379, -0.5122926, -0.43606478, 6.4333878, -1.3908641, -0.48224172, -0.9318897, -1.1649382]]}, {'Basket': [[3.537952, -0.11874729, -0.6143042, 4.150231, -1.6462238, 0.49634176, -1.2271496, -1.7730664, 0.10150815, 0.36229157, -1.1427803, 1.7072755, -0.07331692, -0.8622093], [0.2326435, -0.2758199, -0.67160505, 5.107534, -1.1315546, -0.060734212, -1.6606357, -0.8778804, -0.74439603, -0.7747257, -1.1741363, 3.4446402, -0.66995436, -0.360082]]}]]
    labels_list = ['O', 'B-ACT', 'B-CNT', 'B-OBJ', 'B-OPE', 'B-ORD', 'B-PRE', 'B-TYP', 'B-VAL', 'I-ACT', 'I-CNT', 'I-OBJ', 'I-OPE', 'I-PRE']
    #print(raw_outputs)

    probs = calc_confidence(raw_outputs[0], labels_list)
    print(min(probs))
    print(np.mean(probs))