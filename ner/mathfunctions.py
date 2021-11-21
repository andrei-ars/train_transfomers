import numpy as np

def softmax(X, theta=1):
    # theta - determinism parameter
    ps = np.exp(X * theta)
    ps /= np.sum(ps)
    return ps

def softmax_probabilities(X):
    if type(X) is list:
        X = np.array(X)
    ps = softmax(X).tolist()
    ps = [round(x, 4) for x in ps]
    return ps

if __name__ == "__main__":

    ls = [ 4.5420566 , -2.3695154 , -3.3213828 , -2.1788623 , -1.4054103 , -2.7363303 , 
            -1.817739  , -1.9751874 , -2.7331548 ]
    ps = softmax_probabilities(ls)
    print(ps)