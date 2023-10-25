# pip install POT, pyemd
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
stop_words = stopwords.words('english')

import gensim.downloader
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')
print('[glove vectors] loaded')

def _wmd_distance(sentence_1, sentence_2):
    sentence_1 = sentence_1.lower().split()
    sentence_2 = sentence_2.lower().split()

    sentence_1 = [w for w in sentence_1 if w not in stop_words]
    sentence_2 = [w for w in sentence_2 if w not in stop_words]

    return glove_vectors.wmdistance(sentence_1, sentence_2)


import numpy as np
from pycocoevalcap.custom_eval import _refs_dict

def _wmd_on_image(refs, pred):
    score= [_wmd_distance(ref, pred) for ref in refs]
    return np.mean(score)

def wmd_on_image(image_id, pred, dtype='nocaps'):
    refs = _refs_dict[dtype][image_id]
    if not isinstance(pred, str): pred=pred[0]
    return _wmd_on_image(refs, pred)

def wmd_on_dataset(src, dst, dtype='nocaps'):
    scores=[]
    for s,d in tqdm(zip(src,dst),total=len(src)):
        scores.append(wmd_on_image(s,d,dtype))
    return np.mean(scores)

def wmd_for_pair(text1, text2):
    return _wmd_distance(text1, text2)
