import pandas as pd
import argparse
import os
import json
import numpy as np


def build_text_list(nsample=None):
    if True:
        annotation_file='nocaps_val_4500_captions.json'
        data = json.load(open(annotation_file, 'r'))

        im = pd.DataFrame(data['images'])
        an = pd.DataFrame(data['annotations'])
        an['file_name'] = an.image_id.map(dict(zip(im['id'], im['file_name'])))

        df = pd.DataFrame(data['images'])
        filenames = df.file_name.values.tolist()

        an=an.set_index('file_name')['caption']
        captions = [np.random.choice(an.loc[file]) for file in filenames]

        image_folder = '/home/ubuntu/datasets/nocaps/val/'
        filenames = [image_folder + file for file in filenames]

        if nsample:
            permutation = np.random.permutation(len(filenames))
            filenames=np.array(filenames)[permutation[:nsample]].tolist()
            captions=np.array(captions)[permutation[:nsample]].tolist()
            json.dump([captions, filenames], open('annotation/sd_nocaps_%d.json'%nsample,'w'))
        else:
            json.dump([captions, filenames], open('annotation/sd_nocaps.json','w'))

    if True:
        annotation_file='annotation/coco_karpathy_test.json'
        data = json.load(open(annotation_file, 'r'))
        image_folder='/home/ubuntu/datasets/coco/images/'
        filenames = [image_folder + file["image"] for file in data]
        captions = [np.random.choice(file['caption']) for file in data]

        if nsample:
            permutation = np.random.permutation(len(filenames))
            filenames=np.array(filenames)[permutation[:nsample]].tolist()
            captions=np.array(captions)[permutation[:nsample]].tolist()
            json.dump([captions, filenames], open('annotation/sd_coco_%d.json'%nsample,'w'))
        else:
            json.dump([captions, filenames], open('annotation/sd_coco.json','w'))


def load_text_list(dtype='nocaps', rebuild=False, debug=False):
    if rebuild:
        build_text_list()
        build_text_list(10)

    if dtype=='nocaps':
        captions, filenames = json.load(open('annotation/sd_nocaps.json','r'))
    else:
        captions, filenames = json.load(open('annotation/sd_coco.json','r'))

    if debug:
        if dtype=='nocaps':
            captions, filenames = json.load(open('annotation/sd_nocaps_%d.json'%debug,'r'))
        else:
            captions, filenames = json.load(open('annotation/sd_coco_%d.json'%debug,'r'))
    print(len(captions))
    return captions, filenames
