import os
import json
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import torch
import string
import random
import pandas as pd
from scipy.spatial.distance import cdist
from transformers import CLIPTokenizer, CLIPModel

from pycocoevalcap.custom_eval import _refs_dict


os.environ["TOKENIZERS_PARALLELISM"] = "false"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


device=torch.device('cuda')
clip_model.to(device)

def clip_embedding(text):
    inputs = clip_processor(text=text,return_tensors="pt", padding=True, max_length=clip_processor.model_max_length, truncation=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = clip_model.text_model(**inputs)

    feats_vis=outputs['pooler_output'].cpu().data.numpy()
    return dict(feats_vis=feats_vis)

def _clip_text_on_image(refs, pred):
    src_embeds = clip_embedding(refs)
    dst_embeds = clip_embedding(pred)
    
    fea='feats_vis'
    metric='cosine'

    score=cdist(src_embeds[fea],dst_embeds[fea],metric).flatten()

    return np.mean(score)

def clip_text_on_image(image_id, pred, dtype='nocaps'):
    refs = _refs_dict[dtype][image_id]
    if not isinstance(pred, list): pred=[pred]
    return _clip_text_on_image(refs, pred)


def clip_text_on_dataset(src, dst, dtype='nocaps'):
    scores=[]
    for s,d in tqdm(zip(src,dst),total=len(src)):
        scores.append(clip_text_on_image(s,d, dtype))
    return np.mean(scores)

def clip_text_for_pair(text1, text2):
    assert isinstance(text1, str) and isinstance(text2, str)
    return _clip_text_on_image([text1], [text2])


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sentence_model.to(device)

def bert_on_image(image_id, pred, dtype='nocaps'):
    refs = _refs_dict[dtype][image_id]
    if not isinstance(pred, list): pred=[pred]

    embeds = sentence_model.encode(refs+pred)
    src=embeds[:len(refs)]
    dst=embeds[len(refs):]

    score=cosine_similarity(src, dst)
    return 1-np.mean(score)

def bert_on_dataset(src, dst, dtype='nocaps'):
    scores=[]
    for s,d in tqdm(zip(src,dst),total=len(src)):
        scores.append(bert_on_image(s,d, dtype))
    return np.mean(scores)

def bert_for_pair(text1, text2):
    assert isinstance(text1, str) and isinstance(text2, str)
    embeds = sentence_model.encode([text1, text2])
    src=embeds[:1]
    dst=embeds[1:]
    score=cosine_similarity(src, dst)
    return 1-np.mean(score)
