import os
import json
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import torch
import string
import random
import pandas as pd
from transformers import CLIPImageProcessor, CLIPModel
from scipy.spatial.distance import cdist
from torchvision import transforms


#clip
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

device=torch.device('cuda')
clip_model.to(device)
clip_model.eval()

def clip_embedding(image_path):
    image = Image.open(image_path)
    inputs=clip_processor(image,return_tensors='pt')
    inputs=inputs.to(device)
    with torch.no_grad():
        outputs = clip_model.vision_model(**inputs)

    feats_vis=outputs['pooler_output'].cpu().data.numpy().flatten()
    return dict(feats_vis=feats_vis)

def clip_vision_on_image(src_image, dst_image):
    src_embeds = clip_embedding(src_image)
    dst_embeds = clip_embedding(dst_image)
    
    fea='feats_vis'
    metric='cosine'

    score=cdist(src_embeds[fea][np.newaxis,:],dst_embeds[fea][np.newaxis,:],metric).flatten()[0]

    return score

def clip_vision_on_dataset(src, dst):
    scores=[]
    for s,d in tqdm(zip(src,dst),total=len(src)):
        scores.append(clip_vision_on_image(s,d))
    return np.mean(scores)

def clip_vision_on_dataset_detailed(src, dst):
    scores=[]
    for s,d in tqdm(zip(src,dst),total=len(src)):
        scores.append(clip_vision_on_image(s,d))
    return np.mean(scores), scores



vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
vitb8.to(device)
vitb8.eval()


image_size=384
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_dino_image(filepath):
    img = Image.open(filepath)
    img = img.convert('RGB')
    image = transform(img).unsqueeze(0).to(device)
    return image

def dino_on_image(image1, image2):
    with torch.no_grad():
        emb1=vitb8(load_dino_image(image1)).cpu().numpy()
        emb2=vitb8(load_dino_image(image2)).cpu().numpy()

    metric='cosine'

    score=cdist(emb1,emb2,metric).flatten()[0]
    return score

def dino_on_dataset(src, dst):
    scores=[]
    for s,d in tqdm(zip(src,dst),total=len(src)):
        scores.append(dino_on_image(s,d))
    return np.mean(scores)

def dino_on_dataset_detailed(src, dst):
    scores=[]
    for s,d in tqdm(zip(src,dst),total=len(src)):
        scores.append(dino_on_image(s,d))
    return np.mean(scores), scores
