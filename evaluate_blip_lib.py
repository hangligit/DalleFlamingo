import os
import pandas as pd
import argparse
import json
from tqdm.auto import tqdm

from generation_blip_lib import flamingo, flamingo_batch
from generation_blip_data_lib import create_loader

import sys;sys.path.insert(0,'libs')
from pycocoevalcap.custom_eval_dataset import metric_on_dataset


def load_image_list(datatype='nocaps'):
    if datatype=='nocaps':
        annotation_file = 'nocaps_val_4500_captions.json'
        image_folder = '/home/ubuntu/datasets/nocaps/val/'
        data = json.load(open(annotation_file, 'r'))
        df = pd.DataFrame(data['images'])
        filenames = df.file_name.values.tolist()
        filenames = [image_folder + file for file in filenames]
    else:
        if datatype=='coco_val':
            annotation_file = 'annotation/coco_karpathy_val.json'
        elif datatype=='coco_test':
            annotation_file = 'annotation/coco_karpathy_test.json'
        image_folder = '/home/ubuntu/datasets/coco/images/'
        data = json.load(open(annotation_file, 'r'))
        filenames = [image_folder + file["image"] for file in data]
    return filenames

def evaluate_blip(blip, datatype='nocaps', output_file='result.json',debug=False):
    blip.eval()
    image_files = load_image_list(datatype)
    if debug: image_files=image_files[:10]

    predictions = []
    for image in tqdm(image_files):
        generated_text=flamingo(blip, image)
        predictions.append((os.path.basename(image), generated_text))
    json.dump(predictions, open(output_file, 'w'))
    blip.train()

    coco_eval_result = metric_on_dataset(predictions)
    for metric, score in coco_eval_result.items():
        print(f'{metric}: {score:.3f}')
    output_result_file=output_file.replace('.json','_score.json')
    json.dump(coco_eval_result, open(output_result_file, 'w'))
    return coco_eval_result


def evaluate_blip_batch(blip, datatype='nocaps', output_file='result.json', batch_size=32, num_workers=4, debug=False):
    blip.eval()

    loader=create_loader(batch_size, num_workers, datatype)
    predictions = []
    for image, image_file in tqdm(loader):
        caption=flamingo_batch(blip, image)
        predictions.extend(list(zip(image_file,caption)))
        if debug: break
    json.dump(predictions, open(output_file, 'w'))
    blip.train()

    coco_eval_result = metric_on_dataset(predictions, datatype)
    for metric, score in coco_eval_result.items():
        print(f'{metric}: {score:.3f}')
    output_result_file=output_file.replace('.json','_score.json')
    json.dump(coco_eval_result, open(output_result_file, 'w'))
    return coco_eval_result


def evaluate_blip2(blip, datatype='nocaps', output_file='result.json', batch_size=8, num_workers=4, debug=False):
    blip.eval()

    loader=create_loader(batch_size, num_workers, datatype, image_size=364)
    predictions = []
    for image, image_file in tqdm(loader):
        caption=blip.generate({'image':image.cuda()})
        predictions.extend(list(zip(image_file,caption)))
        if debug: break
    json.dump(predictions, open(output_file, 'w'))
    blip.train()

    coco_eval_result = metric_on_dataset(predictions, datatype)
    for metric, score in coco_eval_result.items():
        print(f'{metric}: {score:.3f}')
    output_result_file=output_file.replace('.json','_score.json')
    json.dump(coco_eval_result, open(output_result_file, 'w'))
    return coco_eval_result

if __name__=='__main__':
    from generation_blip_lib import load_blip
    blip=load_blip()
    evaluate_blip_batch(blip,'coco_test', debug=False)
