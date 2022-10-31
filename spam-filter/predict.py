import numpy as np
from preprocessing import make_bag, to_one_hot, oha_to_text, clean_line

import pickle

clf = pickle.load(open('data/pickles/classifier.pkl', 'rb'))
bow = pickle.load(open('data/pickles/bow.pkl', 'rb'))

def predict(txt):
    txt                 = clean_line(txt)
    oha_txt             = to_one_hot(txt, add_to_bag=False, bow=bow)
    prediction_array    = np.array(oha_txt)
    return clf.predict([prediction_array]) # 1 or 0