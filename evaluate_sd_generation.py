import pandas as pd
import argparse
import os
import json
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from generation_sd_lib import load_sd, dalle, dalle_continue
from evaluate_sd_lib import load_text_list

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--model_id', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--unet', type=str, default=None)
    parser.add_argument('--dtype', type=str, default="nocaps")
    parser.add_argument('--rebuild', action="store_true")
    parser.add_argument('--debug', type=int, default=1000)
    parser.add_argument('--output_file', type=str, default='./image.json')
    parser.add_argument('--use_oneflow', action='store_true')
    parser.add_argument('--skip_generation', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
text_files, image_files=load_text_list(args.dtype, rebuild=args.rebuild, debug=args.debug)

text_files=text_files*5
image_files=image_files*5


if not args.skip_generation:
    sd=load_sd(args.model_id, args.unet, args.fp16)
    if args.resume:
        output = dalle_continue(sd, text_files, args.output_file)
    else:
        output = dalle(sd, text_files, args.output_file)

prediction = json.load(open(args.output_file, 'r'))
prediction = [i[2] for i in prediction]


import sys;sys.path.insert(0,'/home/ubuntu/DalleFlamingo/libs/')
from image_sim_fid import fidelity_on_dataset
from image_sim_embed import clip_vision_on_dataset_detailed

clipdist,clipdist_detailed=clip_vision_on_dataset_detailed(prediction, image_files)

metric=dict(
    fidelity=fidelity_on_dataset(prediction, image_files),
    clipdist=clipdist
)

print(metric)
output_result_file=args.output_file.replace('.json', '_score.json')
json.dump(metric, open(output_result_file, 'w'))

output_result_file=args.output_file.replace('.json', '_detailed_score.json')
json.dump([clipdist_detailed, prediction, image_files, text_files], open(output_result_file, 'w'))

