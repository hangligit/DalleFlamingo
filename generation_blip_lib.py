from PIL import Image
import ruamel.yaml as yaml

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip import blip_decoder
from utils_model import load_model

device = torch.device('cuda')

image_size=384
transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def load_blip(config='/home/ubuntu/DalleFlamingo/configs/caption_coco.yaml', blip_text_decoder=None):
    config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    blip = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                        vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                        prompt=config['prompt'])

    if blip_text_decoder:
        try:
            blip=load_model(blip, blip_text_decoder)
            print('load blip partial from', blip_text_decoder)
        except:
            blip.text_decoder.load_state_dict(torch.load(blip_text_decoder))
            print('load blip text decoder from', blip_text_decoder)
    blip.to(device)
    blip.eval()
    return blip

def flamingo(blip, image,
                sample=False, 
                num_beams=3, 
                max_length=20, 
                min_length=5, 
                repetition_penalty=1.0, 
                top_p=0.9, 
                top_k=None, 
                temperature=None, 
                num_return_sequences=1,
            ):
    """
    if sample, num_beams is set to 1 by default; if not sample, top_p top_k temperature are all ignored
    examples:
        flamingo(image) #beam
        flamingo(image, sample=True) #top_p
        flamingo(image, sample=True, top_p=0.8) #top_p
        flamingo(image, sample=True, top_p=None, top_k=50) #top_k
        flamingo(image, sample=True, top_p=None, temperature=0.7) #temperature
    """
    if sample: repetition_penalty=1.1 #adhere to the original code

    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = blip.generate(
            image, 
            sample=sample,
            num_beams=num_beams,
            max_length=max_length, 
            min_length=min_length, 
            repetition_penalty=repetition_penalty, 
            top_p=top_p, top_k=top_k, 
            temperature=temperature, 
            num_return_sequences=num_return_sequences
        )

    return caption[0]


def flamingo_batch(blip, image,
                sample=False, 
                num_beams=3, 
                max_length=20, 
                min_length=5, 
                repetition_penalty=1.0, 
                top_p=0.9, 
                top_k=None, 
                temperature=None, 
                num_return_sequences=1,
            ):
    """
    if sample, num_beams is set to 1 by default; if not sample, top_p top_k temperature are all ignored
    examples:
        flamingo(image) #beam
        flamingo(image, sample=True) #top_p
        flamingo(image, sample=True, top_p=0.8) #top_p
        flamingo(image, sample=True, top_p=None, top_k=50) #top_k
        flamingo(image, sample=True, top_p=None, temperature=0.7) #temperature
    """
    if sample: repetition_penalty=1.1 #adhere to the original code

    with torch.no_grad():
        caption = blip.generate(
            image.to(device), 
            sample=sample,
            num_beams=num_beams,
            max_length=max_length, 
            min_length=min_length, 
            repetition_penalty=repetition_penalty, 
            top_p=top_p, top_k=top_k, 
            temperature=temperature, 
            num_return_sequences=num_return_sequences
        )

    return caption


if __name__=='__main__':
    blip=load_blip()
    print(flamingo(blip, 'output.png',True))
