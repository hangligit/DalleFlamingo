import pandas as pd
import argparse
import os
import json
from tqdm.auto import tqdm

from generation_blip_lib import load_blip, flamingo

import sys;sys.path.insert(0,'libs')
from pycocoevalcap.custom_eval_dataset import metric_on_dataset, detailed_cider_on_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for BLIP.")
    parser.add_argument('--annotation_file', default='nocaps_val_4500_captions.json')
    parser.add_argument('--image_folder', default='/home/ubuntu/datasets/nocaps/val/')
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--blip_text_decoder', type=str, default='')
    parser.add_argument('--output_file', type=str, default='./result.json')
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--use_beam', type=int, default=1)
    parser.add_argument('--p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--skip_generation', action="store_true")
    parser.add_argument('--dtype', type=str, default='nocaps', help='This is for the new version')
    parser.add_argument('--detailed_metric', action="store_true")
    args = parser.parse_args()
    return args


def load_image_list(annotation_file, image_folder):
    try: #nocaps
        data = json.load(open(annotation_file, 'r'))
        df = pd.DataFrame(data['images'])
        filenames = df.file_name.values.tolist()
        filenames = [image_folder + file for file in filenames]
    except: #coco
        data = json.load(open(annotation_file, 'r'))
        filenames = [image_folder + file["image"] for file in data]
    return filenames


args = parse_args()
image_files = load_image_list(args.annotation_file, args.image_folder)

if not args.skip_generation:
    blip = load_blip(config=args.config, blip_text_decoder=args.blip_text_decoder)

    predictions = []
    for x in tqdm(image_files):
        generated_text=flamingo(blip, x,
                                sample=args.use_beam==0,
                                num_beams=args.num_beams,
                                top_p=args.p,
                                repetition_penalty=args.repetition_penalty,
                            )
        predictions.append((os.path.basename(x), generated_text))
    json.dump(predictions, open(args.output_file, 'w'))

predictions = json.load(open(args.output_file, 'r'))

if not args.detailed_metric:
    coco_eval_result = metric_on_dataset(predictions, args.dtype)
    for metric, score in coco_eval_result.items():
        print(f'{metric}: {score:.3f}')
    output_result_file=args.output_file.replace('.json','_score.json')
    json.dump(coco_eval_result, open(output_result_file, 'w'))
else:
    coco_eval_result = detailed_cider_on_dataset(predictions, args.dtype)
    print(coco_eval_result[0])
    output_result_file=args.output_file.replace('.json','_score_detailed.json')
    json.dump(coco_eval_result, open(output_result_file, 'w'))
