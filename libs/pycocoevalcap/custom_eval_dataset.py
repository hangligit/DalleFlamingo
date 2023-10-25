import pandas as pd
import argparse
import os
import json
import numpy as np

# import sys;sys.path.remove(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))+'/')

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import tempfile

root_dir=os.path.abspath(os.path.join(__file__, os.pardir))


def _load_evaluation_dataset(predictions, dtype='nocaps'):
    """return formatted ground truth and predictions for coco evaluation"""

    if dtype=='nocaps':
        annotation_file = root_dir+'/nocaps_val_4500_captions.json'
        data = json.load(open(annotation_file, 'r'))
        an = pd.DataFrame(data['annotations'])
        im = pd.DataFrame(data['images'])
        an['file_name'] = an.image_id.map(dict(zip(im['id'], im['file_name'])))
        im2id = dict(zip(im['file_name'], im['id']))
    else: #coco
        if dtype=='coco_test':
            annotation_file = root_dir+'/coco_karpathy_test.json'
        elif dtype=='coco_val':
            annotation_file = root_dir+'/coco_karpathy_val.json'
        else:
            annotation_file = root_dir+'/coco_karpathy_full.json'
        data = json.load(open(annotation_file, 'r'))
        an=pd.DataFrame(data)
        an['file_name']=an.image.apply(lambda x: x.split('/')[-1])
        an=an.reset_index().rename({'index':'image_id'},axis=1)
        im2id = dict(zip(an['file_name'], an['image_id']))

    pred = []
    for i, item in enumerate(predictions):
        pred.append({'image_id': int(im2id[item[0]]), 'caption': item[1]})

    imgIds = [i['image_id'] for i in pred]
    ref = dict(
        annotations=[],
        images=[],
    )

    for id in imgIds:
        ref['images'].append({'id': id})
        selected=an[an.image_id == id].caption.values
        if isinstance(selected[0],list):
            selected=selected[0]
        for c in selected:
            ref['annotations'].append({'image_id': id, 'caption': c, 'id': len(ref['annotations'])})

    return ref, pred

def metric_on_dataset(pred, dtype='nocaps'):
    """pred: list of captions, e.g., [(filename.jpg, 'a caption'),...]"""
    assert isinstance(pred, list)
    ref, pred = _load_evaluation_dataset(pred, dtype=dtype)

    annotation_handler=tempfile.NamedTemporaryFile('w',delete=False)
    results_handler=tempfile.NamedTemporaryFile('w',delete=False)

    annotation_file=annotation_handler.name
    results_file=results_handler.name

    json.dump(ref, annotation_handler)
    json.dump(pred, results_handler)

    annotation_handler.close()
    results_handler.close()


    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()

    coco_eval.evaluate()

    os.unlink(annotation_handler.name) # Delete the file
    os.unlink(results_handler.name) # Delete the file

    return coco_eval.eval

def detailed_cider_on_dataset(pred, dtype='nocaps'):
    """pred: list of captions, e.g., [(filename.jpg, 'a caption'),...]"""
    assert isinstance(pred, list)
    ref, pred = _load_evaluation_dataset(pred, dtype=dtype)

    annotation_handler=tempfile.NamedTemporaryFile('w',delete=False)
    results_handler=tempfile.NamedTemporaryFile('w',delete=False)

    annotation_file=annotation_handler.name
    results_file=results_handler.name

    json.dump(ref, annotation_handler)
    json.dump(pred, results_handler)

    annotation_handler.close()
    results_handler.close()


    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()

    coco_eval.evaluate()

    os.unlink(annotation_handler.name) # Delete the file
    os.unlink(results_handler.name) # Delete the file

    if dtype=='nocaps':
        annotation_file = root_dir+'/nocaps_val_4500_captions.json'
        data = json.load(open(annotation_file, 'r'))
        an = pd.DataFrame(data['annotations'])
        im = pd.DataFrame(data['images'])
        an['file_name'] = an.image_id.map(dict(zip(im['id'], im['file_name'])))
        id2im = dict(zip(im['id'],im['file_name']))
    else: #coco
        annotation_file = root_dir+'/coco_karpathy_val.json'
        data = json.load(open(annotation_file, 'r'))
        an=pd.DataFrame(data)
        an['file_name']=an.image.apply(lambda x: x.split('/')[1])
        an=an.reset_index().rename({'index':'image_id'},axis=1)
        id2im = dict(zip( an['image_id'],an['file_name']))

    image_to_metric={id2im[i['image_id']]:i for i in coco_eval.evalImgs}

    detailed_cider=[
        np.mean([image_to_metric[k]['CIDEr'] for k in im[im.domain=='in-domain'].file_name.tolist()]),
        np.mean([image_to_metric[k]['CIDEr'] for k in im[im.domain=='near-domain'].file_name.tolist()]),
        np.mean([image_to_metric[k]['CIDEr'] for k in im[im.domain=='out-domain'].file_name.tolist()]),
    ]
    
    detailed_spice=[
        np.mean([image_to_metric[k]['SPICE']['All']['f'] for k in im[im.domain=='in-domain'].file_name.tolist()]),
        np.mean([image_to_metric[k]['SPICE']['All']['f'] for k in im[im.domain=='near-domain'].file_name.tolist()]),
        np.mean([image_to_metric[k]['SPICE']['All']['f'] for k in im[im.domain=='out-domain'].file_name.tolist()]),
    ]

    scores = [
        {'entire': coco_eval.eval},
        {'in-domain': {'CIDEr':detailed_cider[0],'SPICE':detailed_spice[0]},},
        {'near-domain': {'CIDEr':detailed_cider[1],'SPICE':detailed_spice[1]},},
        {'out-domain': {'CIDEr':detailed_cider[2],'SPICE':detailed_spice[2]},}
    ]

    return scores, image_to_metric
