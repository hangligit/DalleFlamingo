import os
import uuid
import json
import numpy as np
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from utils_model import load_model
import torch

MODELID="CompVis/stable-diffusion-v1-4"

def load_sd(model_id=MODELID, unet_ckpt=None, fp16=False):
    device = "cuda"
    if model_id!=MODELID:
        print('load model v0...')
        torch_dtype=torch.float16 if fp16 else torch.float32
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch_dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODELID,
            unet=unet,
            torch_dtype=torch_dtype)
        pipe=pipe.to(device)
    else:
        if fp16:
            pipe=StableDiffusionPipeline.from_pretrained(MODELID, torch_dtype=torch.float16,)
        else:
            pipe=StableDiffusionPipeline.from_pretrained(MODELID)
        pipe=pipe.to(device)
        if unet_ckpt:
            load_model(pipe.unet, unet_ckpt)
            print('load unet partial from', unet_ckpt)

    return pipe

def dalle_single(sd, prompt, outdir='./'):
    image = sd(prompt, guidance_scale=7.5)['images'][0]
    filename=str(uuid.uuid4())+'.jpg'
    image.save(outdir+'/'+filename, 'JPEG')
    return filename, outdir+'/'+filename

def dalle(sd, prompt_list, output_file='image.json', outdir='/home/ubuntu/assets/generation/'):
    os.makedirs(outdir,exist_ok=True)
    output=[]
    for i, prompt in enumerate(prompt_list):
        image, image_dir=dalle_single(sd, prompt, outdir)
        output.append((prompt, image, image_dir))
        if i%10==0:
            json.dump(output, open(output_file,'w'))
    json.dump(output, open(output_file,'w'))
    return output

def dalle_continue(sd, prompt_list, output_file='image.json', outdir='/home/ubuntu/assets/generation/'):
    os.makedirs(outdir,exist_ok=True)
    assert os.path.isfile(output_file)

    output = json.load(open(output_file))

    json.dump(output, open(output_file.replace('.json','tmpbackup.json'),'w'))

    for i, prompt in enumerate(prompt_list):
        if i<len(output):
            assert output[i][0]==prompt
            assert os.path.isfile(output[i][2])
            continue

        image, image_dir=dalle_single(sd, prompt, outdir)
        output.append((prompt, image, image_dir))
        if i%10==0:
            json.dump(output, open(output_file,'w'))
    json.dump(output, open(output_file,'w'))
    return output
