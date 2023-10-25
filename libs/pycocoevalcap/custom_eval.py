from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .ciderD.cider import Cider as CiderD
from .spice.spice import Spice

import os
root_dir=os.path.abspath(os.path.join(__file__, os.pardir))


def _get_references():
    import json
    import pandas as pd

    refs_dict=dict()

    #nocaps
    annotation_file = root_dir+'/nocaps_val_4500_captions.json'
    data = json.load(open(annotation_file, 'r'))
    an = pd.DataFrame(data['annotations'])
    im = pd.DataFrame(data['images'])
    an['file_name'] = an.image_id.map(dict(zip(im['id'], im['file_name'])))
    filename_to_caption_dict=an.groupby('file_name').agg(list)['caption'].to_dict()
    refs_dict['nocaps']=filename_to_caption_dict

    #coco
    annotation_file = root_dir+'/coco_karpathy_val.json'
    data = json.load(open(annotation_file, 'r'))
    an=pd.DataFrame(data)
    an['file_name']=an['image'].apply(lambda x:x.split('/')[1])
    filename_to_caption_dict=an.set_index('file_name').to_dict()['caption']
    refs_dict['coco']=filename_to_caption_dict
    
    return refs_dict

_refs_dict=_get_references()


import pickle as pkl
_nocaps_df=pkl.load(open(root_dir+'/evalcap_nocaps_df.pkl','rb'))
_nocaps_df_count=4500

_coco_df=pkl.load(open(root_dir+'/evalcap_coco_df.pkl','rb'))
_coco_df_count=40504


def _with_refs_on_image(refs, pred, scorers):
    if isinstance(pred, str): pred=[pred]
    assert isinstance(refs, list)
    assert isinstance(pred, list)

    gts={0:[]}
    for j,ref in enumerate(refs):
        gts[0].append({'image_id':0,'caption':ref,'id':j})

    res={0:[{'image_id':0, 'caption':pred[0],'id':1}]}

    tokenizer = PTBTokenizer(verbose=False)
    gts  = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    score_dict={}
    for scorer, method in scorers:
        # print('computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(gts, res, verbose=0)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                score_dict[m]=sc
        else:
            score_dict[method]=score
    return score_dict


def with_refs_on_image(refs, pred, dtype='nocaps'):
    """
    input:
        refs: a list of reference captions,
        pred: a string of predicted caption,
        dtype: either nocaps or coco

    return:
        a scalar
    """
    df_mode = _nocaps_df if dtype=='nocaps' else _coco_df
    df_count = _nocaps_df_count if dtype=='nocaps' else _coco_df_count
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(df_mode=df_mode, df_count=df_count), "CIDEr"),
        (CiderD(df_mode=df_mode, df_count=df_count), "CIDErD"),
        # (Meteor(),"METEOR"),
        # (Rouge(), "ROUGE_L"),
        # (Spice(), "SPICE")
    ]
    return _with_refs_on_image(refs, pred, scorers)


def retrieve_refs_on_image(img_id, pred, dtype='nocaps'):
    """
    input:
        pred: tuple(image_id, a caption)
        dtype: either nocaps or coco
    return:
        a scalar
    """
    refs=_refs_dict[dtype][img_id]
    score=with_refs_on_image(refs, pred, dtype)
    return score
